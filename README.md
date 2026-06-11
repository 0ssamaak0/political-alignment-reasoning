# Political Alignment and Reasoning

Code for a master's thesis studying how deliberately inducing a political alignment (left or right) in open-weight LLMs affects their reasoning. Alignment is induced with three methods: roleplaying prompts, activation steering, and LoRA fine-tuning.

## Layout

- `politune_hf_train_native/` LoRA fine-tuning of politically aligned models
- `persona_vectors/` extraction of contrastive left/right political persona vectors
- `steering/` activation steering experiments
- `political_compass/` Political Compass Test scoring of the aligned models
- `benchmarking/` reasoning and bias benchmarks (classic evals, G&K bias assessment, custom suites)
- `Judge/` LLM judge used to score model outputs
- `RQ1/`, `RQ2/`, `RQ3/` analysis and figures per research question
