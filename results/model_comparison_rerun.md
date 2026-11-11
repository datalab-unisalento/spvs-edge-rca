# LLM Comparison — full configuration (mean ± std over trials)

- Model A: **gemma-4-e4b** (5 trials)
- Model B: **qwen3.5-9b-q4_k_m** (5 trials)
- Ground truth and baseline are computed from the **full** context (fixed reference).

| Metric | Model A | Model B |
|---|---|---|
| Fine accuracy | 81.88%±1.78% | 72.81%±2.37% |
| Fine macro F1 | 40.89%±0.46% | 26.34%±0.71% |
| Fine balanced accuracy | 45.67%±0.68% | 27.07%±0.77% |
| Fine kappa | 0.695±0.027 | 0.542±0.034 |
| Macro accuracy | 95.31%±1.10% | 79.06%±2.37% |
| Macro F1 | 78.97%±1.92% | 41.26%±1.07% |
| Baseline majority fine accuracy | 51.56%±0.00% | 51.56%±0.00% |
| Baseline majority macro accuracy | 51.56%±0.00% | 51.56%±0.00% |
| Baseline decision tree fine accuracy | 98.44%±0.00% | 98.44%±0.00% |
| Baseline decision tree macro accuracy | 98.44%±0.00% | 98.44%±0.00% |
| Baseline decision tree fine macro F1 | 74.63%±0.00% | 74.63%±0.00% |
| Baseline decision tree macro macro F1 | 66.17%±0.00% | 66.17%±0.00% |
| Baseline decision tree fine kappa | 0.972±0.000 | 0.972±0.000 |
| Avg prompt tokens | 3381±0 | 3328±0 |
| Median prompt tokens | 3400±0 | 3344±0 |
| Avg prompt chars | 6489±0 | 6489±0 |
| Failed calls | 0.0±0.0 | 0.0±0.0 |

### gemma-4-e4b — per trial

| Trial | Fine acc. | Fine macro F1 | Fine kappa | Macro acc. | Macro F1 | Macro precision | Macro recall | Macro bal. acc. | Failed calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 81.25% | 40.81% | 0.686 | 95.31% | 78.41% | 75.00% | 96.97% | 96.97% | 0 |
| 2 | 79.69% | 40.25% | 0.661 | 93.75% | 77.30% | 73.92% | 95.96% | 95.96% | 0 |
| 3 | 81.25% | 40.81% | 0.686 | 95.31% | 78.41% | 75.00% | 96.97% | 96.97% | 0 |
| 4 | 84.38% | 41.52% | 0.733 | 95.31% | 78.41% | 75.00% | 96.97% | 96.97% | 0 |
| 5 | 82.81% | 41.09% | 0.708 | 96.88% | 82.29% | 77.78% | 97.98% | 97.98% | 0 |

### qwen3.5-9b-q4_k_m — per trial

| Trial | Fine acc. | Fine macro F1 | Fine kappa | Macro acc. | Macro F1 | Macro precision | Macro recall | Macro bal. acc. | Failed calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 71.88% | 26.12% | 0.529 | 78.12% | 40.93% | 44.74% | 40.15% | 40.15% | 0 |
| 2 | 76.56% | 27.45% | 0.596 | 82.81% | 42.92% | 45.27% | 42.42% | 42.42% | 0 |
| 3 | 73.44% | 26.50% | 0.549 | 79.69% | 41.50% | 44.74% | 40.91% | 40.91% | 0 |
| 4 | 70.31% | 25.51% | 0.505 | 76.56% | 40.01% | 44.23% | 39.39% | 39.39% | 0 |
| 5 | 71.88% | 26.12% | 0.529 | 78.12% | 40.93% | 44.74% | 40.15% | 40.15% | 0 |

## Inference performance (from LM Studio logs — full config)

- Full-config events: gemma-4-e4b=320, qwen3.5-9b-q4_k_m=320

| Stat | Model A | Model B |
|---|---|---|
| Prompt tokens | 3380.73±87.48 | 3328.02±85.09 |
| Predicted tokens | 204.79±51.82 | 280.65±43.52 |
| Total tokens | 3585.53±123.54 | 3608.66±115.36 |
| Tokens per second | 37.74±1.76 | 23.98±0.28 |
| Time to first token (s) | 1.80±0.11 | 3.07±0.08 |
| Total time (s) | 5.39±1.36 | 11.66±1.81 |
| Confidence | 0.91±0.05 | 0.93±0.08 |
| Failed calls | 0.00±0.00 | 0.00±0.00 |
