# political_compass

Political Compass Test scoring for the induced-alignment configurations: economic
and social coordinates via the PoliLean pipeline, plus a paraphrase-robustness
sweep and the figure scripts.

## Third-party code not included here

This directory's own code (`run_polilean.py`, `run_paraphrase_sweep.py`,
`score_paraphrase_sweep.py`, the `make_*.py` plotters) drives two upstream
projects that we do not redistribute. Obtain them from their sources:

- **PoliLean** (MIT, Shangbin Feng 2024) provides the `step0`-`step3` scripts
  and the political statement set used here. `run_polilean.py` expects it at
  `./PoliLean/`, so clone it into this directory before running:
  - Repo: https://github.com/BunsenFeng/PoliLean
  - Paper: From Pretraining Data to Language Models to Downstream Tasks, ACL 2023, https://arxiv.org/abs/2305.08283

- **Röttger et al., "Political Compass or Spinning Arrow?"** is the literature
  reference behind `spinning_arrow/`. Only our provenance file
  `spinning_arrow/metadata.json` is kept; the original code and paper are at:
  - Repo: https://github.com/paul-rottger/llm-values-pct
  - Paper: arXiv:2402.16786
