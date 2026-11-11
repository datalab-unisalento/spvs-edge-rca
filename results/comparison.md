# Root-Cause LLM Ablation Study (mean ± std over repeated trials)

Each configuration is evaluated on repeated independent LLM inference passes.
Metrics are the mean over trials with the sample standard deviation as the error.
Ground truth and baseline are computed from the **full** context (fixed reference).

| Config | Description | Trials | Fine acc. | Fine macro F1 | Fine bal. acc. | Fine kappa | Macro acc. | Macro F1 | Baseline maj. fine | Baseline maj. macro | Baseline DT fine | Baseline DT macro | Avg prompt tok | Med prompt tok | Avg chars | Failed calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | full context (original system) | 5 | 81.88%±1.78% | 40.89%±0.46% | 45.67%±0.68% | 0.695±0.027 | 95.31%±1.10% | 78.97%±1.92% | 51.56%±0.00% | 51.56%±0.00% | 98.44%±0.00% | 98.44%±0.00% | 3381±0 | 3400±0 | 6489±0 | 0.0±0.0 |
| no_hints | all but inference_hints | 5 | 79.06%±1.40% | 29.26%±0.29% | 28.17%±0.51% | 0.647±0.020 | 98.44%±0.00% | 50.00%±0.00% | 51.56%±0.00% | 51.56%±0.00% | 98.44%±0.00% | 98.44%±0.00% | 3370±0 | 3380±0 | 6448±0 | 0.0±0.0 |
| no_weather | all but history.weather / history.weather_latest | 5 | 77.19%±0.86% | 42.56%±0.36% | 47.08%±0.35% | 0.633±0.012 | 93.12%±0.86% | 64.78%±0.26% | 51.56%±0.00% | 51.56%±0.00% | 98.44%±0.00% | 98.44%±0.00% | 2354±0 | 2372±0 | 5078±0 | 0.0±0.0 |
| no_baseline | all but history.baseline_same_hour | 5 | 83.75%±1.78% | 48.74%±5.95% | 54.31%±6.72% | 0.726±0.028 | 97.50%±0.86% | 84.73%±3.33% | 51.56%±0.00% | 51.56%±0.00% | 98.44%±0.00% | 98.44%±0.00% | 3261±0 | 3292±0 | 6327±0 | 0.0±0.0 |
| minimal | only event + detection_context | 5 | 63.44%±0.86% | 29.50%±0.18% | 26.24%±0.33% | 0.455±0.009 | 68.12%±0.86% | 39.17%±0.22% | 51.56%±0.00% | 51.56%±0.00% | 98.44%±0.00% | 98.44%±0.00% | 678±0 | 729±0 | 2758±0 | 0.0±0.0 |

## Details per configuration

### full — full context (original system)

| Trial | Status | Fine acc. | Fine macro F1 | Fine kappa | Macro acc. | Macro F1 | Macro precision | Macro recall | Macro bal. acc. | Failed calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | inferred | 81.25% | 40.81% | 0.686 | 95.31% | 78.41% | 75.00% | 96.97% | 96.97% | 0 |
| 2 | inferred | 79.69% | 40.25% | 0.661 | 93.75% | 77.30% | 73.92% | 95.96% | 95.96% | 0 |
| 3 | inferred | 81.25% | 40.81% | 0.686 | 95.31% | 78.41% | 75.00% | 96.97% | 96.97% | 0 |
| 4 | inferred | 84.38% | 41.52% | 0.733 | 95.31% | 78.41% | 75.00% | 96.97% | 96.97% | 0 |
| 5 | inferred | 82.81% | 41.09% | 0.708 | 96.88% | 82.29% | 77.78% | 97.98% | 97.98% | 0 |

- Trials: 5
- llm_fine_accuracy: 81.88%±1.78%
- llm_fine_macro_f1: 40.89%±0.46%
- llm_fine_balanced_accuracy: 45.67%±0.68%
- llm_fine_kappa: 0.695±0.027
- llm_macro_accuracy: 95.31%±1.10%
- llm_macro_macro_f1: 78.97%±1.92%
- baseline_majority_fine_accuracy: 51.56%±0.00%
- baseline_majority_macro_accuracy: 51.56%±0.00%
- baseline_dtree_fine_accuracy: 98.44%±0.00%
- baseline_dtree_macro_accuracy: 98.44%±0.00%
- baseline_dtree_fine_macro_f1: 74.63%±0.00%
- baseline_dtree_macro_macro_f1: 66.17%±0.00%
- avg_prompt_tokens: 3381±0
- median_prompt_tokens: 3400±0
- avg_prompt_chars: 6489±0
- failed_calls: 0.0±0.0
- Total failed calls (sum across trials): 0

### no_hints — all but inference_hints

| Trial | Status | Fine acc. | Fine macro F1 | Fine kappa | Macro acc. | Macro F1 | Macro precision | Macro recall | Macro bal. acc. | Failed calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | inferred | 78.12% | 29.07% | 0.634 | 98.44% | 50.00% | 50.00% | 50.00% | 50.00% | 0 |
| 2 | inferred | 78.12% | 29.07% | 0.634 | 98.44% | 50.00% | 50.00% | 50.00% | 50.00% | 0 |
| 3 | inferred | 81.25% | 29.74% | 0.679 | 98.44% | 50.00% | 50.00% | 50.00% | 50.00% | 0 |
| 4 | inferred | 78.12% | 29.07% | 0.634 | 98.44% | 50.00% | 50.00% | 50.00% | 50.00% | 0 |
| 5 | inferred | 79.69% | 29.34% | 0.655 | 98.44% | 50.00% | 50.00% | 50.00% | 50.00% | 0 |

- Trials: 5
- llm_fine_accuracy: 79.06%±1.40%
- llm_fine_macro_f1: 29.26%±0.29%
- llm_fine_balanced_accuracy: 28.17%±0.51%
- llm_fine_kappa: 0.647±0.020
- llm_macro_accuracy: 98.44%±0.00%
- llm_macro_macro_f1: 50.00%±0.00%
- baseline_majority_fine_accuracy: 51.56%±0.00%
- baseline_majority_macro_accuracy: 51.56%±0.00%
- baseline_dtree_fine_accuracy: 98.44%±0.00%
- baseline_dtree_macro_accuracy: 98.44%±0.00%
- baseline_dtree_fine_macro_f1: 74.63%±0.00%
- baseline_dtree_macro_macro_f1: 66.17%±0.00%
- avg_prompt_tokens: 3370±0
- median_prompt_tokens: 3380±0
- avg_prompt_chars: 6448±0
- failed_calls: 0.0±0.0
- Total failed calls (sum across trials): 0

### no_weather — all but history.weather / history.weather_latest

| Trial | Status | Fine acc. | Fine macro F1 | Fine kappa | Macro acc. | Macro F1 | Macro precision | Macro recall | Macro bal. acc. | Failed calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | inferred | 76.56% | 42.33% | 0.624 | 93.75% | 64.97% | 62.50% | 71.74% | 71.74% | 0 |
| 2 | inferred | 76.56% | 42.14% | 0.623 | 92.19% | 64.50% | 62.50% | 70.91% | 70.91% | 0 |
| 3 | inferred | 78.12% | 42.61% | 0.645 | 93.75% | 64.97% | 62.50% | 71.74% | 71.74% | 0 |
| 4 | inferred | 76.56% | 42.64% | 0.625 | 92.19% | 64.50% | 62.50% | 70.91% | 70.91% | 0 |
| 5 | inferred | 78.12% | 43.08% | 0.646 | 93.75% | 64.97% | 62.50% | 71.74% | 71.74% | 0 |

- Trials: 5
- llm_fine_accuracy: 77.19%±0.86%
- llm_fine_macro_f1: 42.56%±0.36%
- llm_fine_balanced_accuracy: 47.08%±0.35%
- llm_fine_kappa: 0.633±0.012
- llm_macro_accuracy: 93.12%±0.86%
- llm_macro_macro_f1: 64.78%±0.26%
- baseline_majority_fine_accuracy: 51.56%±0.00%
- baseline_majority_macro_accuracy: 51.56%±0.00%
- baseline_dtree_fine_accuracy: 98.44%±0.00%
- baseline_dtree_macro_accuracy: 98.44%±0.00%
- baseline_dtree_fine_macro_f1: 74.63%±0.00%
- baseline_dtree_macro_macro_f1: 66.17%±0.00%
- avg_prompt_tokens: 2354±0
- median_prompt_tokens: 2372±0
- avg_prompt_chars: 5078±0
- failed_calls: 0.0±0.0
- Total failed calls (sum across trials): 0

### no_baseline — all but history.baseline_same_hour

| Trial | Status | Fine acc. | Fine macro F1 | Fine kappa | Macro acc. | Macro F1 | Macro precision | Macro recall | Macro bal. acc. | Failed calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | inferred | 81.25% | 40.71% | 0.684 | 96.88% | 82.29% | 77.78% | 97.98% | 97.98% | 0 |
| 2 | inferred | 85.94% | 54.41% | 0.760 | 96.88% | 82.29% | 77.78% | 97.98% | 97.98% | 0 |
| 3 | inferred | 84.38% | 49.74% | 0.732 | 98.44% | 88.38% | 83.33% | 98.99% | 98.99% | 0 |
| 4 | inferred | 84.38% | 54.08% | 0.737 | 98.44% | 88.38% | 83.33% | 98.99% | 98.99% | 0 |
| 5 | inferred | 82.81% | 44.79% | 0.715 | 96.88% | 82.29% | 77.78% | 97.98% | 97.98% | 0 |

- Trials: 5
- llm_fine_accuracy: 83.75%±1.78%
- llm_fine_macro_f1: 48.74%±5.95%
- llm_fine_balanced_accuracy: 54.31%±6.72%
- llm_fine_kappa: 0.726±0.028
- llm_macro_accuracy: 97.50%±0.86%
- llm_macro_macro_f1: 84.73%±3.33%
- baseline_majority_fine_accuracy: 51.56%±0.00%
- baseline_majority_macro_accuracy: 51.56%±0.00%
- baseline_dtree_fine_accuracy: 98.44%±0.00%
- baseline_dtree_macro_accuracy: 98.44%±0.00%
- baseline_dtree_fine_macro_f1: 74.63%±0.00%
- baseline_dtree_macro_macro_f1: 66.17%±0.00%
- avg_prompt_tokens: 3261±0
- median_prompt_tokens: 3292±0
- avg_prompt_chars: 6327±0
- failed_calls: 0.0±0.0
- Total failed calls (sum across trials): 0

### minimal — only event + detection_context

| Trial | Status | Fine acc. | Fine macro F1 | Fine kappa | Macro acc. | Macro F1 | Macro precision | Macro recall | Macro bal. acc. | Failed calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | inferred | 62.50% | 29.30% | 0.445 | 67.19% | 38.93% | 50.00% | 33.56% | 33.56% | 0 |
| 2 | inferred | 64.06% | 29.63% | 0.462 | 68.75% | 39.34% | 50.00% | 34.32% | 34.32% | 0 |
| 3 | inferred | 62.50% | 29.30% | 0.445 | 67.19% | 38.93% | 50.00% | 33.56% | 33.56% | 0 |
| 4 | inferred | 64.06% | 29.63% | 0.462 | 68.75% | 39.34% | 50.00% | 34.32% | 34.32% | 0 |
| 5 | inferred | 64.06% | 29.63% | 0.462 | 68.75% | 39.34% | 50.00% | 34.32% | 34.32% | 0 |

- Trials: 5
- llm_fine_accuracy: 63.44%±0.86%
- llm_fine_macro_f1: 29.50%±0.18%
- llm_fine_balanced_accuracy: 26.24%±0.33%
- llm_fine_kappa: 0.455±0.009
- llm_macro_accuracy: 68.12%±0.86%
- llm_macro_macro_f1: 39.17%±0.22%
- baseline_majority_fine_accuracy: 51.56%±0.00%
- baseline_majority_macro_accuracy: 51.56%±0.00%
- baseline_dtree_fine_accuracy: 98.44%±0.00%
- baseline_dtree_macro_accuracy: 98.44%±0.00%
- baseline_dtree_fine_macro_f1: 74.63%±0.00%
- baseline_dtree_macro_macro_f1: 66.17%±0.00%
- avg_prompt_tokens: 678±0
- median_prompt_tokens: 729±0
- avg_prompt_chars: 2758±0
- failed_calls: 0.0±0.0
- Total failed calls (sum across trials): 0

