/* ============================================================================
   app.js — bootstrap: load manifest, wire tabs / glossary / theme / hash,
   plus App.goto for tour deep links and the share button.
   ========================================================================== */
(function () {
  "use strict";
  const App = window.App;

  function nudge(charts) {
    setTimeout(function () { (charts || []).forEach(function (c) { if (c) try { c.resize(); } catch (e) {} }); }, 30);
  }

  function showMode(mode) {
    App.state.mode = mode;
    App.qs("#view-tour").classList.toggle("hidden", mode !== "tour");
    App.qs("#view-browser").classList.toggle("hidden", mode !== "browser");
    App.qs("#view-strength").classList.toggle("hidden", mode !== "strength");
    Array.prototype.forEach.call(document.querySelectorAll("#tabs .tab"), function (t) {
      t.setAttribute("aria-pressed", String(t.dataset.mode === mode));
    });
    if (mode === "tour") {
      if (!App.Tour._inited) { App.Tour._inited = true; App.Tour.init(); }
    } else if (mode === "browser") {
      if (!App.Browser._inited) { App.Browser._inited = true; App.Browser.init(); }
      else nudge(App.Browser._charts);
    } else {
      if (!App.Strength._inited) { App.Strength._inited = true; App.Strength.init(); }
      else nudge([App.Strength._chart, App.Strength._compareChart]);
    }
    App.saveHash();
  }
  App.showMode = showMode;

  /* Jump to a view with a prepared state. Used by the Findings Tour. */
  App.goto = function (mode, payload) {
    if (mode === "browser") {
      const st = App.Browser.defaultState();
      if (payload) {
        if (payload.filters) Object.assign(st.filters, payload.filters);
        for (const k in payload) if (k !== "filters") st[k] = payload[k];
      }
      App.state.browser = st;
      if (App.Browser._inited) { App.Browser.renderControls(); App.Browser.reload(); }
    } else if (mode === "strength") {
      const st = Object.assign(App.Strength.defaultState(), payload || {});
      App.state.strength = st;
      if (App.Strength._inited) App.Strength.load();
    }
    showMode(mode);
    window.scrollTo({ top: 0 });
  };

  function wireTabs() {
    Array.prototype.forEach.call(document.querySelectorAll("#tabs .tab"), function (t) {
      t.addEventListener("click", function () { showMode(t.dataset.mode); });
    });
    App.qs("#shareBtn").addEventListener("click", function () {
      App.saveHash();
      App.copyText(location.href, "Link to this view copied");
    });
  }

  function buildGlossary() {
    const body = App.qs("#glossaryBody");
    App.manifest.glossary.forEach(function (g) {
      body.appendChild(App.h("div", { class: "gloss-item" }, [
        App.h("div", { class: "gloss-term", text: g.term }),
        App.h("div", { class: "gloss-def", text: g.definition }),
        App.h("div", { class: "gloss-src", text: g.source }),
      ]));
    });
    function open() { App.qs("#glossaryDrawer").classList.remove("hidden"); App.qs("#drawerScrim").classList.remove("hidden"); }
    function close() { App.qs("#glossaryDrawer").classList.add("hidden"); App.qs("#drawerScrim").classList.add("hidden"); }
    App.qs("#glossaryBtn").addEventListener("click", open);
    App.qs("#glossaryClose").addEventListener("click", close);
    App.qs("#drawerScrim").addEventListener("click", close);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") close(); });
  }

  function applyHash() {
    const data = App.readHash();
    if (!data) return;
    App.withHashLock(function () {
      if (data.b) App.state.browser = data.b;
      if (data.s) App.state.strength = data.s;
      if (data.m === "browser" || data.m === "strength" || data.m === "tour") App.state.mode = data.m;
    });
  }

  function boot() {
    App.initTheme();
    App.fetchJSON("data/manifest.json").then(function (manifest) {
      App.manifest = manifest;
      App.buildVocab(manifest);
      if (manifest.title) {
        const bt = App.qs(".brand-title"); // keep short brand, full title in tooltip
        if (bt) bt.title = manifest.title;
      }
      applyHash();
      wireTabs();
      buildGlossary();
      App.themeTick();
      App.qs("#loading").classList.add("hidden");
      showMode(App.state.mode || "tour");
    }).catch(function (e) {
      const l = App.qs("#loading");
      l.textContent = "Failed to load data: " + e.message + "  (serve over http, e.g. python3 -m http.server)";
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
