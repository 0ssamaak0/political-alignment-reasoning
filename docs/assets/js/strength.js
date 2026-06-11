/* ============================================================================
   strength.js — Strength Explorer mode (RQ3).
   Model x Method x Direction -> strength slider (with play) -> trait/coherence/
   accuracy/collapse/contamination curves with the deployed point and the
   collapse zone marked, a signed both-directions panel, and (DPO) per-question
   responses that change with strength.
   ========================================================================== */
(function () {
  "use strict";
  const App = window.App;
  const h = App.h, clear = App.clear;

  const S = (App.Strength = {});
  S._payload = null;
  S._chart = null;
  S._compareChart = null;
  S._timer = null;
  S._playBtn = null;

  S.defaultState = function () { return { model: "mistral", method: "steering", direction: "left", step: null }; };

  S.stopPlay = function () {
    if (S._timer) { clearInterval(S._timer); S._timer = null; }
    if (S._playBtn) S._playBtn.textContent = "▶";
  };

  // chart annotations: collapse zone + deployed strength under the data,
  // slider marker on top
  const markerPlugin = {
    id: "strengthMarker",
    beforeDatasetsDraw: function (chart) {
      const p = S._payload;
      if (!p || !chart.scales.x) return;
      const a = chart.chartArea, ctx = chart.ctx;
      const ci = cliffIndex(p.stops);
      if (ci >= 0) {
        let last = ci;
        for (let i = ci; i < p.stops.length; i++) if (p.stops[i].collapse != null) last = i;
        const x0 = chart.scales.x.getPixelForValue(p.stops[ci].strength);
        const x1 = chart.scales.x.getPixelForValue(p.stops[last].strength);
        ctx.save();
        ctx.globalAlpha = 0.08; ctx.fillStyle = App.cssVar("--purple");
        ctx.fillRect(Math.min(x0, x1), a.top, Math.max(3, Math.abs(x1 - x0)), a.bottom - a.top);
        ctx.restore();
      }
      const dx = chart.scales.x.getPixelForValue(p.deployed);
      if (dx >= a.left && dx <= a.right) {
        ctx.save();
        ctx.strokeStyle = App.cssVar("--text-faint"); ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(dx, a.top); ctx.lineTo(dx, a.bottom); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = App.cssVar("--text-faint");
        ctx.font = "10px " + (App.cssVar("--sans") || "sans-serif");
        ctx.textAlign = "left";
        ctx.fillText("deployed", dx + 4, a.top + 10);
        ctx.restore();
      }
    },
    afterDraw: function (chart) {
      if (chart.$markerX == null || !chart.scales.x) return;
      const px = chart.scales.x.getPixelForValue(chart.$markerX);
      const a = chart.chartArea, ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = App.cssVar("--accent"); ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
      ctx.restore();
    },
  };

  // faint deployed lines on the signed both-directions chart
  const compareDeployedPlugin = {
    id: "compareDeployed",
    beforeDatasetsDraw: function (chart) {
      if (!chart.$deployedXs || !chart.scales.x) return;
      const a = chart.chartArea, ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = App.cssVar("--text-faint"); ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      chart.$deployedXs.forEach(function (x) {
        const px = chart.scales.x.getPixelForValue(x);
        if (px < a.left || px > a.right) return;
        ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
      });
      ctx.restore();
    },
  };

  S.init = function () {
    if (!App.state.strength) App.state.strength = S.defaultState();
    S.renderControls();
    S.load();
  };

  function file() {
    const st = App.state.strength;
    return (((App.manifest.rq3.index[st.model] || {})[st.method] || {})[st.direction]);
  }

  S.renderControls = function () {
    const st = App.state.strength, host = clear(App.qs("#strengthControls"));

    host.appendChild(h("div", { class: "cgroup" }, [h("label", { text: "Model" }),
      seg([{ id: "llama", label: "Llama-3-8B" }, { id: "mistral", label: "Mistral-7B" }], st.model,
        function (id) { st.model = id; st.step = null; S.load(); })]));

    host.appendChild(h("div", { class: "cgroup" }, [h("label", { text: "Method" }),
      seg([{ id: "steering", label: "Activation steering" }, { id: "dpo", label: "DPO fine-tuning" }], st.method,
        function (id) { st.method = id; st.step = null; S.load(); })]));

    host.appendChild(h("div", { class: "cgroup" }, [h("label", { text: "Alignment direction" }),
      seg([{ id: "left", label: "Left", cls: "dir-left" }, { id: "right", label: "Right", cls: "dir-right" }], st.direction,
        function (id) { st.direction = id; S.load(); })]));

    // slider container (filled after load)
    host.appendChild(h("div", { class: "cgroup", id: "sliderGroup" }, [h("label", { text: "Alignment strength" }),
      h("div", { id: "sliderHost" })]));

    host.appendChild(h("p", { class: "panel-note", id: "deployNote" }));
  };

  function seg(items, current, onpick) {
    const wrap = h("div", { class: "seg" });
    items.forEach(function (it) {
      wrap.appendChild(h("button", { class: it.cls || "", "aria-pressed": String(it.id === current),
        onclick: function () { Array.prototype.forEach.call(wrap.children, function (b) { b.setAttribute("aria-pressed", "false"); });
          this.setAttribute("aria-pressed", "true"); onpick(it.id); } }, it.label));
    });
    return wrap;
  }

  S.load = function () {
    S.stopPlay();
    App.saveHash();
    S.renderControls();
    const f = file();
    if (!f) return;
    App.fetchJSON("data/" + f).then(function (p) {
      S._payload = p;
      const st = App.state.strength;
      // default step = deployed if present else middle
      let step = st.step;
      if (step == null || step < 0 || step >= p.stops.length) {
        const di = p.stops.findIndex(function (s) { return s.strength === p.deployed; });
        step = di >= 0 ? di : Math.floor(p.stops.length / 2);
      }
      st.step = step;
      S.buildSlider();
      S.renderAll();
    });
    S.renderCompare();
  };

  function cliffIndex(stops) {
    for (let i = 0; i < stops.length; i++) if (stops[i].collapse != null && stops[i].collapse >= 50) return i;
    return -1;
  }

  S.buildSlider = function () {
    const p = S._payload, st = App.state.strength;
    const host = clear(App.qs("#sliderHost"));
    const stops = p.stops;
    const ci = cliffIndex(stops);

    const valEl = h("div", { class: "slider-val" });
    const playBtn = h("button", { class: "play-btn", title: "Animate the sweep from base to maximum strength", text: "▶" });
    S._playBtn = playBtn;
    const head = h("div", { class: "slider-head" }, [valEl,
      h("span", { class: "slider-side" }, [playBtn, h("span", { class: "panel-note", text: stops.length + " strengths" })])]);
    const range = h("input", { type: "range", class: "strength", min: 0, max: stops.length - 1, step: 1, value: st.step });

    function applyStep() {
      App.saveHash();
      S.renderReadout(); S.moveMarker(); S.renderResponses(); updateVal();
    }
    range.addEventListener("input", function (e) {
      S.stopPlay();
      st.step = +e.target.value;
      applyStep();
    });
    playBtn.addEventListener("click", function () {
      if (S._timer) { S.stopPlay(); return; }
      if (st.step >= stops.length - 1) { st.step = 0; range.value = 0; applyStep(); }
      playBtn.textContent = "⏸";
      S._timer = setInterval(function () {
        if (st.step >= stops.length - 1) { S.stopPlay(); return; }
        st.step++; range.value = st.step;
        applyStep();
      }, 850);
    });

    const ticks = h("div", { class: "ticks" });
    stops.forEach(function (s, i) {
      const cls = "tick" + (s.strength === p.deployed ? " deployed" : "") + (i === ci ? " cliff" : "");
      ticks.appendChild(h("span", { class: cls, text: fmtStrength(s.strength) }));
    });
    function updateVal() {
      const s = stops[st.step];
      valEl.innerHTML = '<span class="unit">' + p.strength_field + " = </span>" + fmtStrength(s.strength)
        + (s.is_base ? '  <span class="unit">(base)</span>' : "");
    }
    updateVal();
    host.appendChild(head); host.appendChild(range); host.appendChild(ticks);

    const dn = App.qs("#deployNote");
    let txt = "Deployed in RQ1 at " + p.strength_field + " = " + fmtStrength(p.deployed) + ".";
    if (ci >= 0) txt += " Coherence cliff near " + p.strength_field + " = " + fmtStrength(stops[ci].strength) + ".";
    if (!p.has_responses) txt += " Per-question responses were not recorded for activation steering.";
    dn.textContent = txt;
  };

  S.renderAll = function () { S.renderReadout(); S.renderChart(); S.renderResponses(); };

  S.renderReadout = function () {
    const p = S._payload, s = p.stops[App.state.strength.step];
    const dirColor = App.state.strength.direction === "left" ? App.cssVar("--left") : App.cssVar("--right");
    const host = clear(App.qs("#strengthReadout"));
    host.appendChild(gauge("Trait score", s.trait, dirColor, "alignment expressed"));
    host.appendChild(gauge("Coherence", s.coherence, App.cssVar("--slate"), "fluency, alignment-blind"));
    host.appendChild(gauge("BBH accuracy", s.accuracy, App.cssVar("--good"), "reparsed, 4 tasks"));
    host.appendChild(gauge("Collapse", s.collapse, App.cssVar("--purple"), "no parseable answer"));
    host.appendChild(gauge("Contamination", s.contam, App.cssVar("--warn"), "added partisan framing"));
  };

  function gauge(k, v, color, sub) {
    const pctw = v == null ? 0 : Math.max(0, Math.min(100, v));
    return h("div", { class: "gauge" }, [
      h("div", { class: "k", text: k }),
      h("div", { class: "v", style: "color:" + (v == null ? "var(--text-faint)" : color), text: v == null ? "—" : App.fmtNum(v) }),
      h("div", { class: "bar" }, [h("span", { style: "width:" + pctw + "%;background:" + color })]),
      h("div", { class: "panel-note", text: sub }),
    ]);
  }

  S.renderChart = function () {
    if (S._chart) { App.destroyCharts([S._chart]); S._chart = null; }
    const p = S._payload;
    const xs = p.stops.map(function (s) { return s.strength; });
    function ds(label, key, color, dash) {
      return { label: label, borderColor: color, backgroundColor: color, tension: 0.25,
        spanGaps: false, borderWidth: 2.5, pointRadius: 3, pointHoverRadius: 5, borderDash: dash || [],
        data: p.stops.map(function (s) { return s[key]; }) };
    }
    const dirColor = App.state.strength.direction === "left" ? App.cssVar("--left") : App.cssVar("--right");
    const canvas = App.qs("#strengthChart");
    const c = new Chart(canvas, {
      type: "line",
      data: { labels: xs, datasets: [
        ds("Trait", "trait", dirColor),
        ds("Coherence", "coherence", App.cssVar("--slate")),
        ds("BBH accuracy", "accuracy", App.cssVar("--good")),
        ds("Collapse", "collapse", App.cssVar("--purple"), [5, 4]),
        ds("Contamination", "contam", App.cssVar("--warn"), [2, 3]),
      ] },
      options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 14, font: { size: 11 } } },
          tooltip: { callbacks: { title: function (items) { return p.strength_field + " = " + items[0].label; },
            label: function (ctx) { return ctx.dataset.label + ": " + (ctx.parsed.y == null ? "n/a" : ctx.parsed.y + "%"); } } } },
        scales: {
          x: { type: "linear", title: { display: true, text: "alignment strength (" + p.strength_field + ")", color: App.cssVar("--text-faint") },
            grid: { color: App.cssVar("--border") }, ticks: { color: App.cssVar("--text-faint") } },
          y: { beginAtZero: true, max: 100, grid: { color: App.cssVar("--border") },
            ticks: { color: App.cssVar("--text-faint"), callback: function (v) { return v + "%"; } } } } },
      plugins: [markerPlugin],
    });
    S._chart = App.trackChart(c);
    S.moveMarker();
    App.qs("#strengthChartNote").textContent = "dashed accent line = slider position";
    App.qs("#strengthLegendHint").textContent =
      "The shaded band is the collapse zone, where half or more of the items return no parseable answer. The faint dotted line marks the strength deployed in RQ1 and RQ2.";
  };

  S.moveMarker = function () {
    if (!S._chart) return;
    S._chart.$markerX = S._payload.stops[App.state.strength.step].strength;
    S._chart.update("none");
  };

  /* ------------------- both directions on one signed axis ------------------- */
  S.renderCompare = function () {
    const st = App.state.strength;
    const idx = ((App.manifest.rq3.index[st.model] || {})[st.method]) || {};
    const panel = App.qs("#comparePanel");
    if (!idx.left || !idx.right) { if (panel) panel.classList.add("hidden"); return; }
    panel.classList.remove("hidden");
    Promise.all([App.fetchJSON("data/" + idx.left), App.fetchJSON("data/" + idx.right)]).then(function (res) {
      const L = res[0], R = res[1];
      if (S._compareChart) { App.destroyCharts([S._compareChart]); S._compareChart = null; }
      function pts(key) {
        const arr = [];
        L.stops.forEach(function (s) { if (s.strength > 0) arr.push({ x: -s.strength, y: s[key] }); });
        const base = R.stops.filter(function (s) { return !s.strength; })[0] || L.stops[0];
        if (base) arr.push({ x: 0, y: base[key] });
        R.stops.forEach(function (s) { if (s.strength > 0) arr.push({ x: s.strength, y: s[key] }); });
        arr.sort(function (a, b) { return a.x - b.x; });
        return arr;
      }
      function ds(label, key, color, dash) {
        return { label: label, borderColor: color, backgroundColor: color, tension: 0.25,
          spanGaps: false, borderWidth: 2.5, pointRadius: 2.5, pointHoverRadius: 5, borderDash: dash || [],
          data: pts(key) };
      }
      const canvas = App.qs("#compareChart");
      const c = new Chart(canvas, {
        type: "line",
        data: { datasets: [
          ds("BBH accuracy", "accuracy", App.cssVar("--good")),
          ds("Collapse", "collapse", App.cssVar("--purple"), [5, 4]),
          ds("Contamination", "contam", App.cssVar("--warn"), [2, 3]),
        ] },
        options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
          plugins: { legend: { position: "bottom", labels: { boxWidth: 14, font: { size: 11 } } },
            tooltip: { callbacks: {
              title: function (items) {
                const x = items[0].parsed.x;
                const side = x < 0 ? "left, " : (x > 0 ? "right, " : "base, ");
                return side + L.strength_field + " = " + Math.abs(x);
              },
              label: function (ctx) { return ctx.dataset.label + ": " + (ctx.parsed.y == null ? "n/a" : ctx.parsed.y + "%"); } } } },
          scales: {
            x: { type: "linear",
              title: { display: true, text: "← aligned left        " + L.strength_field + "        aligned right →", color: App.cssVar("--text-faint") },
              grid: { color: App.cssVar("--border") },
              ticks: { color: App.cssVar("--text-faint"), callback: function (v) { return Math.abs(v); } } },
            y: { beginAtZero: true, max: 100, grid: { color: App.cssVar("--border") },
              ticks: { color: App.cssVar("--text-faint"), callback: function (v) { return v + "%"; } } } } },
        plugins: [compareDeployedPlugin],
      });
      c.$deployedXs = [-L.deployed, R.deployed];
      S._compareChart = App.trackChart(c);
      App.qs("#compareNote").textContent =
        "left direction at negative strength, right at positive, base at zero · dotted lines = deployed strengths";
    });
  };

  S.renderResponses = function () {
    const p = S._payload, s = p.stops[App.state.strength.step];
    const host = clear(App.qs("#strengthResponses"));
    host.appendChild(h("div", { class: "panel-head" }, [
      h("h3", { text: "Model responses at this strength" }),
      h("span", { class: "panel-note", text: p.strength_field + " = " + fmtStrength(s.strength) }),
    ]));
    if (!s.per_question || !s.per_question.length) {
      host.appendChild(h("p", { class: "panel-note", text: p.has_responses
        ? "No per-question responses at the base point."
        : "Per-question responses were not recorded for activation steering. The curves above summarize the probe questions at each strength." }));
      return;
    }
    s.per_question.forEach(function (q) {
      host.appendChild(h("div", { class: "resp-item" }, [
        h("div", { class: "resp-q", text: q.question }),
        h("div", { class: "resp-a", text: q.response }),
        h("div", { class: "resp-scores" }, [
          chip("trait " + App.fmtNum(q.trait), App.state.strength.direction === "left" ? "left" : "right"),
          chip("coherence " + App.fmtNum(q.coherence), q.coherence != null && q.coherence < 40 ? "purple" : "slate"),
        ]),
      ]));
    });
  };

  function chip(text, cls) { return h("span", { class: "tag " + cls, text: text }); }
  function fmtStrength(v) { return v == null ? "—" : (Number.isInteger(v) ? String(v) : String(v)); }
})();
