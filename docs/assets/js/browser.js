/* ============================================================================
   browser.js — Example Browser mode.
   Flow: Benchmark -> Model -> Config -> judge-label filters -> stats + cards.
   ========================================================================== */
(function () {
  "use strict";
  const App = window.App;
  const h = App.h, clear = App.clear;

  const B = (App.Browser = {});
  B._records = [];     // currently loaded cell records
  B._filtered = [];
  B._rendered = 40;
  B._charts = [];
  B._spotlight = null;

  function defaultState() {
    return {
      benchmark: "boolean_expressions",
      model: "llama",
      config: "base",
      filters: {
        outcome: [], contaminated: "any", collapsed: "any",
        category: [], validity: [], fallacy: [],
        confMin: 1, item_lean: [], valid: [],
      },
      search: "", sort: "default",
    };
  }

  B.init = function () {
    if (!App.state.browser) App.state.browser = defaultState();
    B.renderControls();
    B.reload();
  };

  function bench() { return App.benchById[App.state.browser.benchmark]; }
  function isRQ2() { return bench().rq === "rq2"; }
  function rq() { return bench().rq; }

  /* ----------------------------- controls ----------------------------- */
  B.renderControls = function () {
    const st = App.state.browser, host = clear(App.qs("#browserControls"));

    // Benchmark (grouped select)
    const sel = h("select", { class: "input", onchange: function (e) {
      st.benchmark = e.target.value;
      // reset extra-axis filters that don't apply across rq
      st.filters.item_lean = []; st.filters.valid = []; st.filters.confMin = 1;
      B.renderControls(); B.reload();
    }});
    const groups = { neutral: h("optgroup", { label: "Neutral reasoning" }),
                     political: h("optgroup", { label: "Political reasoning" }) };
    App.manifest.benchmarks.forEach(function (b) {
      const o = h("option", { value: b.id, text: b.label + (b.has_examples ? "" : "  (aggregate only)") });
      if (b.id === st.benchmark) o.selected = true;
      groups[b.type].appendChild(o);
    });
    host.appendChild(h("div", { class: "cgroup" }, [h("label", { text: "Benchmark" }), sel,
      groups.neutral, groups.political].slice(0, 2)));
    // append optgroups into the select
    sel.appendChild(groups.neutral); sel.appendChild(groups.political);

    // Model segmented
    host.appendChild(h("div", { class: "cgroup" }, [
      h("label", { text: "Model" }),
      seg(App.manifest.models.map(function (m) { return { id: m.id, label: m.id === "llama" ? "Llama-3-8B" : "Mistral-7B" }; }),
        st.model, function (id) { st.model = id; B.reload(); }),
    ]));

    // Config chips
    const cfgWrap = h("div", { class: "chips" });
    App.manifest.configs.forEach(function (c) {
      const dirClass = "dir-" + (c.direction || "none");
      const chip = h("button", {
        class: "chip config " + dirClass, "aria-pressed": String(c.id === st.config),
        onclick: function () { st.config = c.id; B.renderControls(); B.reload(); },
      }, c.label);
      cfgWrap.appendChild(chip);
    });
    host.appendChild(h("div", { class: "cgroup" }, [h("label", { text: "Configuration" }), cfgWrap]));

    if (!bench().has_examples) {
      host.appendChild(h("p", { class: "panel-note", text: "This benchmark has aggregate stats only." }));
      return;
    }

    // Search
    host.appendChild(h("div", { class: "cgroup" }, [
      h("label", { text: "Search text" }),
      h("input", { class: "search", type: "search", placeholder: "question, response, or judge note…",
        value: st.search, oninput: function (e) { st.search = e.target.value; B.refilter(); } }),
    ]));

    // Filters header with reset
    const fhead = h("div", { class: "cgroup" }, [
      h("div", { style: "display:flex;justify-content:space-between;align-items:center" }, [
        h("span", { class: "clabel", text: "Filters", style: "margin:0" }),
        h("button", { class: "reset-link", text: "reset", onclick: function () {
          const keep = { benchmark: st.benchmark, model: st.model, config: st.config };
          App.state.browser = Object.assign(defaultState(), keep);
          B.renderControls(); B.refilter();
        }}),
      ]),
    ]);
    host.appendChild(fhead);

    // Outcome
    host.appendChild(chipGroup("Outcome", App.manifest.vocab.outcome, st.filters.outcome));
    // Flags (tri-state)
    host.appendChild(triState("Contaminated", "contaminated"));
    host.appendChild(triState("Collapsed", "collapsed"));
    // Primary category
    host.appendChild(chipGroup("Primary category", App.manifest.vocab.primary_category, st.filters.category,
      function (k) { return App.CAT_COLOR[k]; }));
    // Reasoning validity
    host.appendChild(chipGroup("Reasoning validity", App.manifest.vocab.reasoning_validity, st.filters.validity));

    if (isRQ2()) {
      host.appendChild(chipGroup("Argument side", [{ key: "left", label: "Left" }, { key: "right", label: "Right" }],
        st.filters.item_lean, function (k) { return k === "left" ? App.cssVar("--left") : App.cssVar("--right"); }));
      host.appendChild(chipGroup("Argument validity", [{ key: "valid", label: "Valid" }, { key: "invalid", label: "Invalid" }],
        st.filters.valid));
    } else {
      // Confidence min slider
      const v = h("span", { class: "chip-count", text: "≥ " + st.filters.confMin });
      const r = h("input", { type: "range", min: 1, max: 5, step: 1, value: st.filters.confMin,
        class: "strength", oninput: function (e) { st.filters.confMin = +e.target.value; v.textContent = "≥ " + e.target.value; B.refilter(); } });
      host.appendChild(h("div", { class: "cgroup" }, [
        h("div", { style: "display:flex;justify-content:space-between" }, [h("span", { class: "clabel", style: "margin:0", text: "Min confidence" }), v]), r]));
    }
    // Fallacy lens (collapsible-ish, just a chip group)
    host.appendChild(chipGroup("Fallacy lens", App.manifest.vocab.fallacy_lens, st.filters.fallacy));
  };

  function seg(items, current, onpick) {
    const wrap = h("div", { class: "seg" });
    items.forEach(function (it) {
      wrap.appendChild(h("button", { "aria-pressed": String(it.id === current),
        onclick: function () { Array.prototype.forEach.call(wrap.children, function (b) { b.setAttribute("aria-pressed", "false"); });
          this.setAttribute("aria-pressed", "true"); onpick(it.id); } }, it.label));
    });
    return wrap;
  }

  function chipGroup(title, items, arr, colorFn) {
    const wrap = h("div", { class: "chips" });
    items.forEach(function (it) {
      const on = arr.indexOf(it.key) >= 0;
      const dot = colorFn ? h("span", { class: "dot", style: "background:" + colorFn(it.key) }) : null;
      wrap.appendChild(h("button", { class: "chip", "aria-pressed": String(on),
        onclick: function () {
          const i = arr.indexOf(it.key);
          if (i >= 0) arr.splice(i, 1); else arr.push(it.key);
          this.setAttribute("aria-pressed", String(i < 0));
          B.refilter();
        }}, [dot, document.createTextNode(it.label)].filter(Boolean)));
    });
    return h("div", { class: "cgroup" }, [h("label", { text: title }), wrap]);
  }

  function triState(title, key) {
    const st = App.state.browser.filters;
    const opts = [{ id: "any", label: "Any" }, { id: "yes", label: "Yes" }, { id: "no", label: "No" }];
    return h("div", { class: "cgroup" }, [h("label", { text: title }),
      seg(opts, st[key], function (id) { st[key] = id; B.refilter(); })]);
  }

  /* ----------------------------- data load ----------------------------- */
  B.reload = function () {
    App.saveHash();
    const st = App.state.browser;
    if (!bench().has_examples) { B.renderAggregate(); return; }
    const entry = ((((App.manifest.examples[rq()] || {})[st.model] || {})[st.config] || {})[st.benchmark]);
    const statsEl = clear(App.qs("#browserStats"));
    statsEl.appendChild(h("div", { class: "panel", text: "Loading examples…" }));
    clear(App.qs("#browserResults")); clear(App.qs("#resultsBar"));
    if (!entry) { statsEl.firstChild.textContent = "No data for this selection."; return; }
    App.fetchJSON("data/" + entry.file).then(function (records) {
      B._records = records;
      B._cellStats = entry.stats;
      B._rendered = 40; B._spotlight = null;
      B.refilter();
    }).catch(function (e) { statsEl.firstChild.textContent = "Error: " + e.message; });
  };

  B.refilter = function () {
    App.saveHash();
    const st = App.state.browser;
    B._filtered = applyFilters(B._records, st.filters, isRQ2(), st.search);
    B._rendered = 40;
    B.renderStats();
    B.renderResults();
  };

  function applyFilters(records, f, rq2, search) {
    const q = (search || "").trim().toLowerCase();
    return records.filter(function (r) {
      if (f.outcome.length && f.outcome.indexOf(r.outcome) < 0) return false;
      if (f.contaminated === "yes" && !r.contaminated) return false;
      if (f.contaminated === "no" && r.contaminated) return false;
      if (f.collapsed === "yes" && !r.collapsed) return false;
      if (f.collapsed === "no" && r.collapsed) return false;
      if (f.category.length && f.category.indexOf(r.category) < 0) return false;
      if (f.validity.length && f.validity.indexOf(r.validity) < 0) return false;
      if (f.fallacy.length && f.fallacy.indexOf(r.fallacy) < 0) return false;
      if (!rq2 && f.confMin > 1 && (r.confidence == null || r.confidence < f.confMin)) return false;
      if (rq2 && f.item_lean.length && f.item_lean.indexOf(r.item_lean) < 0) return false;
      if (rq2 && f.valid.length && f.valid.indexOf(r.valid) < 0) return false;
      if (q) {
        const hay = ((r.prompt || "") + " " + (r.response || "") + " " + (r.judge_reasoning || "")).toLowerCase();
        if (hay.indexOf(q) < 0) return false;
      }
      return true;
    });
  }

  /* ----------------------------- stats ----------------------------- */
  function liveStats(records) {
    const s = { n: records.length, correct: 0, contaminated: 0, collapsed: 0, noverdict: 0,
      cat: {}, conf: { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 }, sideTot: { left: 0, right: 0 }, sideCorrect: { left: 0, right: 0 } };
    App.CAT_ORDER.forEach(function (k) { s.cat[k] = 0; });
    records.forEach(function (r) {
      if (r.outcome === "correct") s.correct++;
      if (r.contaminated) s.contaminated++;
      if (r.collapsed) s.collapsed++;
      if (r.outcome === "no_answer" || r.outcome === "off_format") s.noverdict++;
      if (s.cat[r.category] != null) s.cat[r.category]++;
      if (r.confidence != null && s.conf[r.confidence] != null) s.conf[r.confidence]++;
      if (r.item_lean) { s.sideTot[r.item_lean]++; if (r.outcome === "correct") s.sideCorrect[r.item_lean]++; }
    });
    return s;
  }

  B.renderStats = function () {
    App.destroyCharts(B._charts); B._charts = [];
    const host = clear(App.qs("#browserStats"));
    const cell = B._cellStats, live = liveStats(B._filtered);
    const filtered = B._filtered.length !== B._records.length;

    // base reference for deltas
    const st = App.state.browser;
    const baseEntry = (((App.manifest.examples[rq()] || {})[st.model] || {}).base || {})[st.benchmark];
    const base = baseEntry ? baseEntry.stats : null;
    const isBase = st.config === "base";

    function card(k, cellVal, baseVal, betterDown) {
      const children = [h("div", { class: "k", text: k }), h("div", { class: "v", text: App.fmtPct(cellVal) })];
      if (base && !isBase && cellVal != null && baseVal != null) {
        const d = Math.round((cellVal - baseVal) * 10) / 10;
        const dir = d === 0 ? "flat" : ((d < 0) === !!betterDown ? "up" : "down");
        children.push(h("div", { class: "delta " + dir, text: (d > 0 ? "+" : "") + d + " vs Base" }));
      } else if (!isBase) {
        children.push(h("div", { class: "delta flat", text: "—" }));
      } else {
        children.push(h("div", { class: "delta flat", text: "reference" }));
      }
      return h("div", { class: "stat-card" }, children);
    }

    const cards = h("div", { class: "stat-cards" }, [
      card("Judge accuracy", cell.judge_acc, base && base.judge_acc, false),
      card("Contamination", cell.contam, base && base.contam, true),
      card("Collapse", cell.collapse, base && base.collapse, true),
      card("No verdict", cell.no_verdict, base && base.no_verdict, true),
    ]);
    host.appendChild(cards);

    const note = h("div", { class: "panel-note", style: "margin:-6px 2px 2px",
      text: "Cards show the full cell (" + cell.n + " items, matching the thesis). "
        + (filtered ? "Filter shows " + live.n + " items." : "No filter applied.") });
    host.appendChild(note);

    // charts row
    const catCanvas = h("canvas");
    const secCanvas = h("canvas");
    const row = h("div", { class: "panels-row" }, [
      h("div", { class: "panel" }, [
        h("div", { class: "panel-head" }, [h("h3", { text: "Failure-mode mix" }),
          h("span", { class: "panel-note", text: filtered ? "filtered subset" : "whole cell" })]),
        h("div", { class: "chart-box" }, catCanvas)]),
      h("div", { class: "panel" }, [
        h("div", { class: "panel-head" }, [h("h3", { text: isRQ2() ? "Accuracy by argument side" : "Confidence" }),
          h("span", { class: "panel-note", text: filtered ? "filtered subset" : "whole cell" })]),
        h("div", { class: "chart-box" }, secCanvas)]),
    ]);
    host.appendChild(row);

    drawCategoryChart(catCanvas, live);
    if (isRQ2()) drawSideChart(secCanvas, live); else drawConfidenceChart(secCanvas, live);
  };

  function drawCategoryChart(canvas, s) {
    const labels = [], data = [], colors = [];
    App.CAT_ORDER.forEach(function (k) { labels.push(App.lbl("primary_category", k)); data.push(s.cat[k]); colors.push(App.CAT_COLOR[k]); });
    const c = new Chart(canvas, {
      type: "bar",
      data: { labels: labels, datasets: [{ data: data, backgroundColor: colors, borderRadius: 4, barThickness: 16 }] },
      options: { indexAxis: "y", responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: function (ctx) { return ctx.parsed.x + " items"; } } } },
        scales: { x: { beginAtZero: true, grid: { color: App.cssVar("--border") }, ticks: { color: App.cssVar("--text-faint") } },
                  y: { grid: { display: false }, ticks: { color: App.cssVar("--text-faint"), font: { size: 11 } } } },
        onClick: function (evt, els) {
          if (!els.length) return;
          const key = App.CAT_ORDER[els[0].index];
          const arr = App.state.browser.filters.category, i = arr.indexOf(key);
          if (i >= 0) arr.splice(i, 1); else arr.push(key);
          B.renderControls(); B.refilter();
        } },
    });
    B._charts.push(App.trackChart(c));
  }

  function drawConfidenceChart(canvas, s) {
    const c = new Chart(canvas, {
      type: "bar",
      data: { labels: ["1", "2", "3", "4", "5"], datasets: [{ data: [s.conf[1], s.conf[2], s.conf[3], s.conf[4], s.conf[5]],
        backgroundColor: App.cssVar("--accent"), borderRadius: 4 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { grid: { display: false }, title: { display: true, text: "judge confidence (1–5)", color: App.cssVar("--text-faint") } },
          y: { beginAtZero: true, grid: { color: App.cssVar("--border") }, ticks: { color: App.cssVar("--text-faint") } } } },
    });
    B._charts.push(App.trackChart(c));
  }

  function drawSideChart(canvas, s) {
    const accL = s.sideTot.left ? 100 * s.sideCorrect.left / s.sideTot.left : 0;
    const accR = s.sideTot.right ? 100 * s.sideCorrect.right / s.sideTot.right : 0;
    const c = new Chart(canvas, {
      type: "bar",
      data: { labels: ["Left argument", "Right argument"], datasets: [{ data: [Math.round(accL * 10) / 10, Math.round(accR * 10) / 10],
        backgroundColor: [App.cssVar("--left"), App.cssVar("--right")], borderRadius: 6, barThickness: 48 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false },
        tooltip: { callbacks: { label: function (ctx) { return ctx.parsed.y + "% correct"; } } } },
        scales: { x: { grid: { display: false } }, y: { beginAtZero: true, max: 100, grid: { color: App.cssVar("--border") },
          ticks: { color: App.cssVar("--text-faint"), callback: function (v) { return v + "%"; } } } } },
    });
    B._charts.push(App.trackChart(c));
  }

  /* ----------------------------- results + cards ----------------------------- */
  B.renderResults = function () {
    const st = App.state.browser;
    const bar = clear(App.qs("#resultsBar"));
    bar.appendChild(h("span", { class: "results-count", html: "<b>" + B._filtered.length + "</b> of " + B._records.length + " examples" }));
    bar.appendChild(h("span", { class: "spacer" }));
    bar.appendChild(h("button", { class: "ghost-btn", text: "Surprise me", onclick: B.surprise }));
    const sortSel = h("select", { class: "mini-select", onchange: function (e) { st.sort = e.target.value; B.renderResults(); } }, [
      h("option", { value: "default", text: "Sort: original order" }),
      !isRQ2() && h("option", { value: "conf", text: "Sort: confidence" }),
      h("option", { value: "tokens", text: "Sort: response length" }),
      h("option", { value: "category", text: "Sort: category" }),
    ].filter(Boolean));
    sortSel.value = st.sort;
    bar.appendChild(sortSel);

    const list = clear(App.qs("#browserResults"));
    let recs = B._filtered.slice();
    if (st.sort === "conf") recs.sort(function (a, b) { return (b.confidence || 0) - (a.confidence || 0); });
    else if (st.sort === "tokens") recs.sort(function (a, b) { return (b.ntok || (b.response || "").length) - (a.ntok || (a.response || "").length); });
    else if (st.sort === "category") recs.sort(function (a, b) { return App.CAT_ORDER.indexOf(a.category) - App.CAT_ORDER.indexOf(b.category); });

    if (!recs.length) { list.appendChild(h("div", { class: "empty", text: "No examples match these filters." })); return; }

    const show = Math.min(B._rendered, recs.length);
    for (let i = 0; i < show; i++) list.appendChild(card(recs[i]));
    if (recs.length > show) {
      list.appendChild(h("button", { class: "ghost-btn", style: "align-self:center;margin-top:8px",
        text: "Show " + Math.min(40, recs.length - show) + " more  (" + (recs.length - show) + " hidden)",
        onclick: function () { B._rendered += 40; B.renderResults(); } }));
    }
    if (B._spotlight != null) {
      const el = list.querySelector('[data-id="' + B._spotlight + '"]');
      if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); el.classList.add("flash"); B._spotlight = null; }
    }
  };

  B.surprise = function () {
    if (!B._filtered.length) return;
    const idx = Math.floor(Math.random() * B._filtered.length);
    B._spotlight = B._filtered[idx].id;
    B._rendered = Math.max(B._rendered, idx + 1);
    // re-sort-agnostic: ensure original-order index visible; simplest is bump render to cover whole list
    B._rendered = B._filtered.length;
    B.renderResults();
  };

  function tag(cls, label, color) {
    const dot = color ? h("span", { class: "dot", style: "background:" + color }) : null;
    return h("span", { class: "tag " + cls }, [dot, document.createTextNode(label)].filter(Boolean));
  }

  function card(r) {
    const rq2 = isRQ2();
    const top = h("div", { class: "card-top" }, [
      h("span", { class: "card-id", text: "#" + r.id }),
      rq2
        ? h("span", { class: "card-ans", html: "gold <b>" + r.valid + "</b> &middot; model <b>" + (r.parsed_verdict || "—") + "</b>" })
        : h("span", { class: "card-ans", html: "gold <b>" + esc(r.gold) + "</b> &middot; parsed <b>" + esc(r.parsed == null ? "—" : r.parsed) + "</b>" }),
      h("span", { class: "spacer" }),
      tag(App.OUTCOME_TAG[r.outcome] || "neutral", App.lbl("outcome", r.outcome)),
      (!rq2 && r.confidence != null) ? stars(r.confidence) : null,
    ].filter(Boolean));

    const tags = h("div", { class: "tags", style: "margin-bottom:10px" }, [
      r.contaminated ? tag("warn", "Contaminated") : null,
      r.collapsed ? tag("purple", "Collapsed") : null,
      tag(App.CAT_TAG[r.category] || "neutral", App.lbl("primary_category", r.category), App.CAT_COLOR[r.category]),
      tag(App.VALIDITY_TAG[r.validity] || "neutral", "Reasoning: " + App.lbl("reasoning_validity", r.validity)),
      (r.fallacy && r.fallacy !== "none") ? tag("slate", App.lbl("fallacy_lens", r.fallacy)) : null,
      rq2 && r.item_lean ? tag(App.DIR_TAG[r.item_lean], (r.item_lean === "left" ? "Left" : "Right") + " argument") : null,
      rq2 ? tag("neutral", "Argument " + r.valid) : null,
    ].filter(Boolean));

    const respText = h("div", { class: "response-text clamped", text: r.response || "(no response)" });
    const moreBtn = h("button", { class: "toggle-more", text: "Show full response", onclick: function () {
      const cl = respText.classList.toggle("clamped");
      this.textContent = cl ? "Show full response" : "Show less";
    }});

    const body = h("div", { class: "card-body" }, [
      tags,
      h("p", { class: "field-label", text: "Prompt" }),
      h("div", { class: "prompt-text", text: r.prompt || "" }),
      h("p", { class: "field-label", text: "Model response" }),
      respText, ((r.response || "").length > 320 ? moreBtn : null),
      h("div", { class: "judge-box" }, [
        h("p", { class: "field-label", text: "Judge" }),
        h("div", { class: "judge-reason", text: r.judge_reasoning || "" }),
        r.justification ? h("div", { class: "judge-quote", text: "“" + r.justification + "”" }) : null,
      ].filter(Boolean)),
    ].filter(Boolean));

    return h("div", { class: "card", dataset: { id: r.id } }, [top, body]);
  }

  function stars(n) {
    const w = h("span", { class: "conf", title: "judge confidence " + n + "/5" });
    for (let i = 1; i <= 5; i++) w.appendChild(document.createTextNode(i <= n ? "★" : "☆"));
    return w;
  }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>]/g, function (c) { return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]; }); }

  /* ----------------------------- party-fixed aggregate ----------------------------- */
  B.renderAggregate = function () {
    App.destroyCharts(B._charts); B._charts = [];
    const st = App.state.browser;
    const statsEl = clear(App.qs("#browserStats"));
    clear(App.qs("#resultsBar"));
    const res = clear(App.qs("#browserResults"));
    statsEl.appendChild(h("div", { class: "note-banner",
      text: "The Party-fixed content-swap test has no per-example judge records. Showing aggregate measures from the thesis." }));

    App.fetchJSON("data/" + App.manifest.rq2_aggregate.party_fixed.file).then(function (agg) {
      const cell = ((agg[st.model] || {})[st.config]);
      if (!cell) { res.appendChild(h("div", { class: "empty", text: "No aggregate for this selection." })); return; }
      const ov = cell.overall || {};
      statsEl.appendChild(h("div", { class: "stat-cards" }, [
        miniStat("Accuracy", App.fmtPct(ov.acc)),
        miniStat("Engaged", (ov.engaged != null ? ov.engaged : "—") + " / " + (ov.n != null ? ov.n : "—")),
        miniStat("False-positive rate", App.fmtPct(ov.fp_rate)),
        miniStat("False-negative rate", App.fmtPct(ov.fn_rate)),
      ]));
      // by-condition table
      const panel = h("div", { class: "panel" }, [h("div", { class: "panel-head" }, [h("h3", { text: "Accuracy by side and condition" }),
        h("span", { class: "panel-note", text: "signed bias is also shown; positive = right skew, negative = left skew" })])]);
      const canvas = h("canvas");
      panel.appendChild(h("div", { class: "chart-box tall" }, canvas));
      res.appendChild(panel);
      const sb = cell.signed_bias || {};
      res.appendChild(h("div", { class: "stat-cards" }, [
        miniStat("Signed bias (all)", fmtSigned(sb.all)),
        miniStat("Signed bias (clean)", fmtSigned(sb.clean)),
        miniStat("Signed bias (flipped)", fmtSigned(sb.flipped)),
        miniStat("Net political belief effect", fmtSigned(cell.net_political_belief_effect)),
      ]));
      drawConditionChart(canvas, cell);
    });
  };

  function drawConditionChart(canvas, cell) {
    const by = cell.by_arm_lean || {};
    const L = by.political_left || {}, R = by.political_right || {};
    function acc(o, k) { return (o[k] && o[k].acc != null) ? o[k].acc : null; }
    const c = new Chart(canvas, {
      type: "bar",
      data: { labels: ["Clean", "Flipped"],
        datasets: [
          { label: "Left argument", data: [acc(L, "clean"), acc(L, "flipped")], backgroundColor: App.cssVar("--left"), borderRadius: 5 },
          { label: "Right argument", data: [acc(R, "clean"), acc(R, "flipped")], backgroundColor: App.cssVar("--right"), borderRadius: 5 },
        ] },
      options: { responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: function (ctx) { return ctx.dataset.label + ": " + ctx.parsed.y + "%"; } } } },
        scales: { x: { grid: { display: false } }, y: { beginAtZero: true, max: 100, grid: { color: App.cssVar("--border") },
          ticks: { color: App.cssVar("--text-faint"), callback: function (v) { return v + "%"; } } } } },
    });
    B._charts.push(App.trackChart(c));
  }

  function miniStat(k, v) { return h("div", { class: "stat-card" }, [h("div", { class: "k", text: k }), h("div", { class: "v", text: v })]); }
  function fmtSigned(v) { if (v == null) return "—"; const x = Math.round(v * 1000) / 1000; return (x > 0 ? "+" : "") + x; }
})();
