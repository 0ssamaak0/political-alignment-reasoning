# RQ3 — MMLU formal logic (strength sweep)

MMLU formal logic scores for the RQ3 strength grid. This is the log-likelihood
benchmark (no generation): the model ranks fixed options A–D and the highest-
probability option is the prediction. Scored 5-shot on the full 126-item test set.

**Margin above chance** = accuracy − 25 (four-choice chance floor). Standard error
at n = 126 is about ±4 percentage points, so small moves near the floor are noisy.

Numbers are read from `mmlu_formal_logic/results.json` in each strength cell.
Regenerate with `python3 RQ3/make_mmlu_table.py --results RQ3/results` when the
local result tree is available (the raw cells are not shipped in this repo; this
file is the canonical published table).

## Mistral-7B-Instruct-v0.2

### Steering coefficient α

**Left alignment**

| Strength | MMLU acc (%) | Margin above chance |
|--:|--:|--:|
| 0 (base) | 38.9 | +13.9 |
| α = 0.5 | 36.5 | +11.5 |
| α = 1 | 35.7 | +10.7 |
| α = 2 | 35.7 | +10.7 |
| α = 3 | 31.7 | +6.7 |
| α = 4 | 27.8 | +2.8 |

**Right alignment**

| Strength | MMLU acc (%) | Margin above chance |
|--:|--:|--:|
| 0 (base) | 38.9 | +13.9 |
| α = 0.5 | 39.7 | +14.7 |
| α = 1 | 41.3 | +16.3 |
| α = 2 | 38.1 | +13.1 |
| α = 3 | 40.5 | +15.5 |
| α = 4 | 34.9 | +9.9 |

### DPO LoRA scale s

**Left alignment**

| Strength | MMLU acc (%) | Margin above chance |
|--:|--:|--:|
| 0 (base) | 38.9 | +13.9 |
| s = 0.25 | 38.9 | +13.9 |
| s = 0.5 | 37.3 | +12.3 |
| s = 1 | 36.5 | +11.5 |
| s = 1.5 | 37.3 | +12.3 |
| s = 2 | 37.3 | +12.3 |

**Right alignment**

| Strength | MMLU acc (%) | Margin above chance |
|--:|--:|--:|
| 0 (base) | 38.9 | +13.9 |
| s = 0.25 | 37.3 | +12.3 |
| s = 0.5 | 37.3 | +12.3 |
| s = 1 | 35.7 | +10.7 |
| s = 1.5 | 34.1 | +9.1 |
| s = 2 | 32.5 | +7.5 |

## Llama-3-8B-Instruct

### Steering coefficient α

**Left alignment**

| Strength | MMLU acc (%) | Margin above chance |
|--:|--:|--:|
| 0 (base) | 37.3 | +12.3 |
| α = 0.5 | 41.3 | +16.3 |
| α = 1 | 38.1 | +13.1 |
| α = 2 | 40.5 | +15.5 |
| α = 3 | 34.9 | +9.9 |
| α = 4 | 28.6 | +3.6 |

**Right alignment**

| Strength | MMLU acc (%) | Margin above chance |
|--:|--:|--:|
| 0 (base) | 37.3 | +12.3 |
| α = 0.5 | 33.3 | +8.3 |
| α = 1 | 31.0 | +6.0 |
| α = 2 | 29.4 | +4.4 |
| α = 3 | 29.4 | +4.4 |
| α = 4 | 30.2 | +5.2 |

### DPO LoRA scale s

**Left alignment**

| Strength | MMLU acc (%) | Margin above chance |
|--:|--:|--:|
| 0 (base) | 37.3 | +12.3 |
| s = 0.25 | 39.7 | +14.7 |
| s = 0.5 | 41.3 | +16.3 |
| s = 1 | 42.9 | +17.9 |
| s = 1.5 | 40.5 | +15.5 |
| s = 2 | 43.7 | +18.7 |

**Right alignment**

| Strength | MMLU acc (%) | Margin above chance |
|--:|--:|--:|
| 0 (base) | 37.3 | +12.3 |
| s = 0.25 | 39.7 | +14.7 |
| s = 0.5 | 37.3 | +12.3 |
| s = 1 | 37.3 | +12.3 |
| s = 1.5 | 29.4 | +4.4 |
| s = 2 | 29.4 | +4.4 |
