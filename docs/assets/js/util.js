/* ============================================================================
   util.js — shared state, vocab, colors, DOM helpers, URL hash, theme.
   Exposes a single global: App
   ========================================================================== */
(function () {
  "use strict";
  const App = (window.App = window.App || {});

  /* ----------------------------- DOM helper ----------------------------- */
  App.h = function (tag, attrs, children) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        const v = attrs[k];
        if (v == null || v === false) continue;
        if (k === "class") el.className = v;
        else if (k === "html") el.innerHTML = v;
        else if (k === "text") el.textContent = v;
        else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
        else if (k === "dataset") { for (const d in v) el.dataset[d] = v[d]; }
        else el.setAttribute(k, v);
      }
    }
    if (children != null) {
      (Array.isArray(children) ? children : [children]).forEach(function (c) {
        if (c == null || c === false) return;
        el.appendChild(typeof c === "string" || typeof c === "number"
          ? document.createTextNode(String(c)) : c);
      });
    }
    return el;
  };
  App.clear = function (node) { while (node.firstChild) node.removeChild(node.firstChild); return node; };
  App.qs = function (sel, root) { return (root || document).querySelector(sel); };

  /* ----------------------------- fetch cache ----------------------------- */
  const _cache = {};
  App.fetchJSON = function (url) {
    if (!_cache[url]) {
      _cache[url] = fetch(url).then(function (r) {
        if (!r.ok) throw new Error("fetch failed: " + url + " (" + r.status + ")");
        return r.json();
      });
    }
    return _cache[url];
  };

  /* ----------------------------- vocab lookups ----------------------------- */
  App.labelMaps = {}; // vocabName -> {key: label}
  App.buildVocab = function (manifest) {
    const v = manifest.vocab;
    ["outcome", "primary_category", "reasoning_validity", "fallacy_lens", "boolean_flags"].forEach(function (name) {
      const map = {};
      (v[name] || []).forEach(function (e) { map[e.key] = e.label; });
      App.labelMaps[name] = map;
    });
    App.configLabel = {}; App.configDir = {};
    manifest.configs.forEach(function (c) { App.configLabel[c.id] = c.label; App.configDir[c.id] = c.direction; });
    App.modelLabel = {}; manifest.models.forEach(function (m) { App.modelLabel[m.id] = m.label; });
    App.methodLabel = {}; manifest.methods.forEach(function (m) { App.methodLabel[m.id] = m.label; });
    App.benchById = {}; manifest.benchmarks.forEach(function (b) { App.benchById[b.id] = b; });
  };
  App.lbl = function (vocab, key) {
    const m = App.labelMaps[vocab];
    return (m && m[key] != null) ? m[key] : key;
  };

  /* ----------------------------- colors ----------------------------- */
  // Precise hex for charts (independent of theme background).
  App.CAT_COLOR = {
    faithful_task_performance: "#16a34a",
    post_hoc_reasoning: "#0d9488",
    capability_error: "#ea580c",
    instruction_following_failure: "#64748b",
    viewpoint_bias: "#dc2626",
    motivational_framing_bias: "#d97706",
    generation_collapse: "#7c3aed",
  };
  App.CAT_ORDER = [
    "faithful_task_performance", "post_hoc_reasoning", "capability_error",
    "instruction_following_failure", "viewpoint_bias", "motivational_framing_bias", "generation_collapse",
  ];
  // Pill style class for tags.
  App.CAT_TAG = {
    faithful_task_performance: "good",
    post_hoc_reasoning: "neutral",
    capability_error: "warn",
    instruction_following_failure: "slate",
    viewpoint_bias: "bad",
    motivational_framing_bias: "purple",
    generation_collapse: "purple",
  };
  App.OUTCOME_TAG = { correct: "good", wrong: "bad", no_answer: "slate", off_format: "neutral" };
  App.VALIDITY_TAG = { valid: "good", invalid: "bad", opaque: "slate", "n/a": "neutral" };
  App.DIR_TAG = { left: "left", right: "right" };

  /* ----------------------------- formatting ----------------------------- */
  App.fmtPct = function (v) { return v == null ? "—" : (Math.round(v * 10) / 10) + "%"; };
  App.fmtNum = function (v) { return v == null ? "—" : (Math.round(v * 10) / 10); };
  App.titleCase = function (s) {
    return String(s || "").replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  };

  /* ----------------------------- state + URL hash ----------------------------- */
  App.state = {
    mode: "browser",
    browser: null,   // set by browser.js on first init
    strength: null,  // set by strength.js on first init
  };
  let _hashLock = false;
  App.saveHash = function () {
    if (_hashLock) return;
    const s = App.state;
    const payload = { m: s.mode, b: s.browser, s: s.strength };
    const str = encodeURIComponent(JSON.stringify(payload));
    history.replaceState(null, "", "#" + str);
  };
  App.readHash = function () {
    const raw = location.hash.replace(/^#/, "");
    if (!raw) return null;
    try { return JSON.parse(decodeURIComponent(raw)); } catch (e) { return null; }
  };
  App.withHashLock = function (fn) { _hashLock = true; try { fn(); } finally { _hashLock = false; } };

  /* ----------------------------- theme ----------------------------- */
  const order = ["auto", "light", "dark"];
  App.initTheme = function () {
    const saved = localStorage.getItem("judgeTheme") || "auto";
    App.applyTheme(saved);
    const btn = App.qs("#themeBtn");
    btn.addEventListener("click", function () {
      const cur = localStorage.getItem("judgeTheme") || "auto";
      const next = order[(order.indexOf(cur) + 1) % order.length];
      App.applyTheme(next);
    });
    // react to OS change when in auto
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
        if ((localStorage.getItem("judgeTheme") || "auto") === "auto") App.themeTick();
      });
    }
  };
  App.applyTheme = function (t) {
    localStorage.setItem("judgeTheme", t);
    document.documentElement.setAttribute("data-theme", t === "auto" ? "auto" : t);
    const btn = App.qs("#themeBtn");
    if (btn) btn.textContent = t.charAt(0).toUpperCase() + t.slice(1);
    App.themeTick();
  };
  // re-theme charts on theme change
  App._charts = [];
  App.cssVar = function (name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  };
  App.themeTick = function () {
    if (window.Chart) {
      Chart.defaults.color = App.cssVar("--text-soft");
      Chart.defaults.borderColor = App.cssVar("--border");
      Chart.defaults.font.family = App.cssVar("--sans") || "sans-serif";
    }
    App._charts.forEach(function (c) {
      if (!c || !c.options) return;
      try {
        if (c.options.scales) for (const ax in c.options.scales) {
          const sc = c.options.scales[ax];
          if (sc.grid) sc.grid.color = App.cssVar("--border");
          if (sc.ticks) sc.ticks.color = App.cssVar("--text-faint");
        }
        c.update("none");
      } catch (e) { /* ignore */ }
    });
  };
  App.trackChart = function (c) { App._charts.push(c); return c; };
  App.destroyCharts = function (list) {
    list.forEach(function (c) {
      const i = App._charts.indexOf(c);
      if (i >= 0) App._charts.splice(i, 1);
      try { c.destroy(); } catch (e) {}
    });
  };
})();
