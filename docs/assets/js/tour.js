/* ============================================================================
   tour.js — Findings Tour mode.
   A guided index of the findings in Chapter 4 of the thesis. Every finding
   links to the records or sweep curves behind it via App.goto, with the
   filters pre-set so the link lands on exactly the evidence described.
   ========================================================================== */
(function () {
  "use strict";
  const App = window.App;
  const h = App.h;

  const T = (App.Tour = {});

  /* Deep-link payload helpers. Strength step is an index into the stop grid:
     steering grid [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5], DPO grid [0, 0.25, 0.5, 1, 1.5, 2]. */
  function b(benchmark, model, config, filters, search) {
    return { mode: "browser", payload: { benchmark: benchmark, model: model, config: config, filters: filters || {}, search: search || "" } };
  }
  function s(model, method, direction, step) {
    return { mode: "strength", payload: { model: model, method: method, direction: direction, step: step } };
  }

  const RQS = [
    {
      kicker: "RQ1 · Method",
      section: "Thesis Section 4.1",
      title: "Which alignment method degrades reasoning the least on neutral tasks?",
      intro: "Seven configurations per model, a base anchor plus a left and a right version of roleplaying, activation steering, and DPO fine-tuning, all matched on alignment strength and run on neutral BBH reasoning tasks. The stat cards in the Example Browser always show the change against the base model.",
      findings: [
        {
          title: "Roleplaying costs the least, steering the most",
          body: "With format noise removed by robust re-parsing, the methods rank from best to worst as <b>roleplaying, then DPO fine-tuning, then activation steering</b>. Roleplaying stays within about a point of the base model, DPO costs little on Llama but more on Mistral, and steering costs the most, with right steering losing the most through collapse.",
          links: [
            { label: "A steering cell next to its base", go: b("boolean_expressions", "mistral", "steer-right") },
            { label: "A roleplay cell next to its base", go: b("boolean_expressions", "mistral", "roleplay-left") },
          ],
        },
        {
          title: "One benchmark score hides three different failures",
          body: "The degradation behind a falling score is not one thing but three. A <b>genuine reasoning loss</b>, an <b>instruction-following tax</b> where the right answer is written in the wrong format, and <b>generation collapse</b> where no answer can be parsed at all. Standard exact-match scoring conflates the three, and on Mistral the recovered format errors alone are large enough to change the method ranking.",
          links: [
            { label: "Genuine wrong reasoning", go: b("web_of_lies", "llama", "steer-left", { category: ["capability_error"] }) },
            { label: "Right answer, wrong format", go: b("boolean_expressions", "mistral", "dpo-left", { outcome: ["off_format"] }) },
            { label: "Collapse, nothing to parse", go: b("boolean_expressions", "mistral", "steer-right", { outcome: ["no_answer"] }) },
          ],
        },
        {
          title: "Political text on neutral tasks is a Mistral steering effect",
          body: "Political content appearing in the reasoning is not a general property of alignment. On the neutral BBH tasks roleplaying and DPO are clean on both models, at <b>0% contamination</b>, and only Mistral steering writes political content into neutral logic items, <b>12%</b> of responses at the left configuration and <b>19%</b> at the right. The thesis quotes a response that reads a political meaning into the logical operator <i>or</i>.",
          links: [
            { label: "Find the quoted response", go: b("boolean_expressions", "mistral", "steer-left", {}, "gender-neutral") },
            { label: "All contaminated responses in that cell", go: b("boolean_expressions", "mistral", "steer-left", { contaminated: "yes" }) },
          ],
        },
      ],
    },
    {
      kicker: "RQ2 · Input type",
      section: "Thesis Section 4.2",
      title: "What does political content in the question do to the reasoning?",
      intro: "Two political benchmarks with fixed logical form. The value-loaded benchmark puts the politics in the conclusion, so the verdict has a political payoff. The party-fixed benchmark keeps the named party fixed and swaps only the policy content, turning a believable partisan premise into a false one.",
      findings: [
        {
          badge: "central finding",
          title: "The Partisan Double Standard",
          body: "Each value-loaded item pairs a left and a right version of the same logical form, so the correct verdict is identical. Mistral DPO-right accepts the valid argument favoring its own side <b>50%</b> of the time, and the logically identical argument favoring the other side only <b>10%</b>. The base models show no gap of their own, Mistral-base sits at a signed bias of exactly <b>0.000</b>.",
          links: [
            { label: "See the gap, valid arguments by side", go: b("value_loaded", "mistral", "dpo-right", { valid: ["valid"] }) },
            { label: "The base treats both sides the same", go: b("value_loaded", "mistral", "base", { valid: ["valid"] }) },
          ],
        },
        {
          title: "A silent skew and a loud one",
          body: "Analyzing the reasoning shows the same right skew reached in two opposite ways. Mistral DPO-right skews <b>silently</b>, adding partisan wording to only about 2% of its responses, while Llama DPO-right skews <b>loudly</b>, with about 98% of responses adding partisan framing as the stated reason for the verdict.",
          links: [
            { label: "The loud case", go: b("value_loaded", "llama", "dpo-right", { contaminated: "yes" }) },
            { label: "The silent case, rejected valid left arguments", go: b("value_loaded", "mistral", "dpo-right", { item_lean: ["left"], valid: ["valid"], outcome: ["wrong"] }) },
          ],
        },
        {
          title: "One-sided refusals can fake a bias",
          body: "Llama roleplay-left leaves <b>66 of its 192 items</b> with no verdict, and <b>55</b> of those refusals fall on the right items. Its large apparent right skew of +0.238 comes from these one-sided refusals, not from judging the two sides differently, so a bias number must always be read together with the engagement rate.",
          links: [
            { label: "Read the refusals", go: b("value_loaded", "llama", "roleplay-left", { outcome: ["no_answer"], item_lean: ["right"] }) },
          ],
        },
        {
          title: "Political content alone makes reasoning worse",
          body: "On the party-fixed benchmark, pooled over all fourteen configurations, every judge measure worsens when political content is added to a question, and worsens again when the partisan premise becomes false. Accuracy falls while contamination, the content-over-form fallacy, and invalid reasoning rise, and the cost appears already at base and under roleplaying, where the model weights are untouched.",
          links: [
            { label: "Accuracy by side and condition", go: b("party_fixed", "mistral", "base") },
          ],
        },
        {
          title: "The skew follows the content, not the party name",
          body: "Where a directional effect appears on the party-fixed benchmark, it flips sign when the policy content flips, which a party-label explanation cannot produce. Mistral steering-right moves from <b>+0.145</b> on believable partisan premises to <b>&minus;0.127</b> on false ones. The effect agrees in sign with the value-loaded skew but rests on little data.",
          links: [
            { label: "The clearest configuration", go: b("party_fixed", "mistral", "steer-right") },
          ],
        },
      ],
    },
    {
      kicker: "RQ3 · Strength",
      section: "Thesis Section 4.3",
      title: "What happens when the alignment is made stronger?",
      intro: "Steering and fine-tuning each have one strength hyperparameter, the steering coefficient α and the DPO adapter scale s. The Strength Explorer sweeps them. Drag the slider, or press play, and watch reasoning hold and then collapse.",
      findings: [
        {
          title: "Reasoning holds, then falls off a cliff",
          body: "Reasoning is not traded smoothly against alignment strength. It holds across a usable range and then drops sharply at a located point, the <b>cliff</b>. On Mistral, steering collapses at <b>α = 4</b> and DPO already at <b>s = 1.5</b>, one step past its trained strength. On Llama only DPO-right reaches a sharp drop, and Llama DPO-left stays robust even at twice the trained strength.",
          links: [
            { label: "Stand on the Mistral steering cliff", go: s("mistral", "steering", "left", 7) },
            { label: "The robust exception, Llama DPO-left", go: s("llama", "dpo", "left", 5) },
          ],
        },
        {
          title: "The collapse is silence, not confident nonsense",
          body: "At the cliff the wrong category empties and the collapse category fills, so the model stops committing to an answer rather than turning to wrong reasoning. Over-driving DPO past its trained scale of 1.0 also buys no stronger alignment, the trait score has already saturated, so everything past that point is pure damage.",
          links: [
            { label: "Watch the failure mode flip at s = 1.5", go: s("mistral", "dpo", "left", 4) },
          ],
        },
        {
          title: "The broken output is political, in the alignment's direction",
          body: "At maximum strength the broken output usually carries political text pointing the way the model was aligned, and which fine-tuned side does this flips between the models, DPO-left on Mistral and DPO-right on Llama. The exception shows the two effects are separate. Mistral DPO-right collapses on <b>71%</b> of items with <b>zero</b> political content. On Mistral, five independent LLM graders re-scored a blind sample and matched the judge on <b>97%</b> of it.",
          links: [
            { label: "Mistral DPO-left over-driven to s = 2", go: s("mistral", "dpo", "left", 5) },
            { label: "Collapse with no politics, Mistral DPO-right", go: s("mistral", "dpo", "right", 5) },
          ],
        },
      ],
    },
  ];

  function linkBtn(l) {
    return h("button", { class: "link-btn", onclick: function () { App.goto(l.go.mode, l.go.payload); } },
      [h("span", { class: "link-arrow", text: "→" }), document.createTextNode(l.label)]);
  }

  function findingCard(f, idx) {
    return h("div", { class: "finding" }, [
      h("div", { class: "finding-head" }, [
        h("span", { class: "fnum", text: String(idx) }),
        h("h3", { text: f.title }),
        f.badge ? h("span", { class: "tag warn fbadge", text: f.badge }) : null,
      ].filter(Boolean)),
      h("p", { class: "finding-body", html: f.body }),
      h("div", { class: "finding-links" }, f.links.map(linkBtn)),
    ]);
  }

  T.init = function () {
    const host = App.qs("#tourBody");

    // hero
    host.appendChild(h("div", { class: "tour-hero" }, [
      h("h1", { text: "Does a political alignment change how a model reasons?" }),
      h("p", { class: "lede", html:
        "The thesis aligns two open models, <b>Llama-3-8B</b> and <b>Mistral-7B</b>, to the left or the right of US politics with three methods, " +
        "roleplaying, activation steering, and DPO fine-tuning on partisan data. An LLM judge then analyzes every response, scoring the reasoning itself and not only the final answer." }),
      h("p", { class: "lede", html:
        "This page is a guided index of the findings in <b>Chapter 4</b>. Every finding links straight to the evidence behind it, " +
        "the judged model responses in the Example Browser or the sweep curves in the Strength Explorer, with the filters already set for you." }),
    ]));

    // map of the app
    host.appendChild(h("div", { class: "tour-map" }, [
      mapCard("Example Browser", "Every judged response on the neutral BBH tasks (RQ1) and the value-loaded benchmark (RQ2), with the judge's reasoning behind each label.",
        function () { App.goto("browser", null); }),
      mapCard("Strength Explorer", "The RQ3 strength sweeps for steering and DPO. Trait, coherence, accuracy, collapse, and contamination at every strength.",
        function () { App.goto("strength", null); }),
      mapCard("Glossary", "The thesis definition of every judge label and metric, with its source chapter. The same definitions appear as tooltips on the tags.",
        function () { App.qs("#glossaryBtn").click(); }),
    ]));

    // two-source hypothesis note
    host.appendChild(h("div", { class: "two-source" }, [
      h("b", { text: "The two-source hypothesis. " }),
      document.createTextNode("Alignment harms reasoning through a general cost of the alignment method, source one, and a separate effect of political content in the question, source two. RQ1 measures source one on neutral tasks, RQ2 tests source two, and RQ3 adds the strength axis."),
    ]));

    // findings by RQ
    let n = 0;
    RQS.forEach(function (rq) {
      host.appendChild(h("div", { class: "rq-head" }, [
        h("div", { class: "rq-kicker" }, [
          h("span", { text: rq.kicker }),
          h("span", { class: "sec-badge", text: rq.section }),
        ]),
        h("h2", { text: rq.title }),
        h("p", { class: "rq-intro", text: rq.intro }),
      ]));
      rq.findings.forEach(function (f) { n++; host.appendChild(findingCard(f, n)); });
    });

    host.appendChild(h("p", { class: "tour-footer", text:
      "Built from the same judge records and sweep files the thesis reports. In the Example Browser the stat cards always show the full cell, matching the thesis tables, and filters only change the charts and the example list below them. The Share view button in the header copies a link to any view, including the filters." }));
  };

  function mapCard(title, body, onclick) {
    return h("button", { class: "map-card", onclick: onclick }, [
      h("div", { class: "map-title", text: title }),
      h("div", { class: "map-body", text: body }),
    ]);
  }
})();
