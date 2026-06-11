# Political Alignment and Reasoning

Code for a master's thesis studying how deliberately inducing a political alignment (left or right) in open-weight LLMs affects their reasoning. Alignment is induced with three methods: roleplaying prompts, activation steering, and LoRA fine-tuning.

## Interactive results explorer

**Live at [0ssamaak0.github.io/political-alignment-reasoning](https://0ssamaak0.github.io/political-alignment-reasoning/)** — an interactive companion to the thesis, built from the same judge records and sweep files the thesis reports.

| Tab | What it shows |
| --- | --- |
| **Findings Tour** | A guided index of the findings in Chapter 4. Every finding links straight to the evidence behind it, with filters pre-set. |
| **Example Browser** | Every judged response on the neutral BBH tasks (RQ1) and the value-loaded benchmark (RQ2), filterable by outcome, judge category, contamination, argument side, and free-text search. Stat cards always show the full cell, matching the thesis tables. |
| **Strength Explorer** | The RQ3 strength sweeps for steering and DPO. Trait, coherence, accuracy, collapse, and contamination at every strength, with the deployed point and the collapse cliff marked, plus per-question responses for DPO. |

Any view can be shared as a link (the URL encodes the full state, filters included), and every judge label carries its thesis definition as a tooltip.

### Findings Tour
![Findings Tour](media/explorer_tour.png)

### Example Browser, the Partisan Double Standard cell
![Example Browser](media/explorer_browser.png)

### Strength Explorer, Mistral DPO-left at the cliff
![Strength Explorer](media/explorer_strength.png)

## Layout

- `politune_hf_train_native/` LoRA fine-tuning of politically aligned models
- `persona_vectors/` extraction of contrastive left/right political persona vectors
- `steering/` activation steering experiments
- `political_compass/` Political Compass Test scoring of the aligned models
- `benchmarking/` reasoning and bias benchmarks (classic evals, G&K bias assessment, custom suites)
- `Judge/` LLM judge used to score model outputs
- `RQ1/`, `RQ2/`, `RQ3/` analysis and figures per research question
- `docs/` the interactive results explorer (served by GitHub Pages), `tools/build_data.py` rebuilds its data
