/* ============================================================================
   app.js — bootstrap: load manifest, wire tabs / glossary / theme / hash.
   ========================================================================== */
(function () {
  "use strict";
  const App = window.App;

  function nudge(charts) {
    setTimeout(function () { (charts || []).forEach(function (c) { if (c) try { c.resize(); } catch (e) {} }); }, 30);
  }

  function showMode(mode) {
    App.state.mode = mode;
    App.qs("#view-browser").classList.toggle("hidden", mode !== "browser");
    App.qs("#view-strength").classList.toggle("hidden", mode !== "strength");
    Array.prototype.forEach.call(document.querySelectorAll("#tabs .tab"), function (t) {
      t.setAttribute("aria-pressed", String(t.dataset.mode === mode));
    });
    if (mode === "browser") {
      if (!App.Browser._inited) { App.Browser._inited = true; App.Browser.init(); }
      else nudge(App.Browser._charts);
    } else {
      if (!App.Strength._inited) { App.Strength._inited = true; App.Strength.init(); }
      else nudge([App.Strength._chart]);
    }
    App.saveHash();
  }

  function wireTabs() {
    Array.prototype.forEach.call(document.querySelectorAll("#tabs .tab"), function (t) {
      t.addEventListener("click", function () { showMode(t.dataset.mode); });
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
      if (data.m === "browser" || data.m === "strength") App.state.mode = data.m;
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
      showMode(App.state.mode || "browser");
    }).catch(function (e) {
      const l = App.qs("#loading");
      l.textContent = "Failed to load data: " + e.message + "  (serve over http, e.g. python3 -m http.server)";
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
