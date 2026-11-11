# Macro-category confusion matrix (full configuration)

Rows are the **reference** macro labels, columns are the **predicted** macro labels.
`false positives` collects the reference `atmospheric_effects` episodes (weather/environmental
effects that are not real faults). `other` collects every other predicted macro label
(e.g. `unknown`). Predicted cells are averaged over the trials, so they can be fractional.

### gemma-4-e4b (mean over 5 trials)

| Reference (macro) | n | physical_electrical | telemetry_and_sensing | atmosphere | other |
|---|---:|---:|---:|---:|---:|
| physical_electrical | 33 | 30 | 0.20 | 2.80 | 0 |
| telemetry_and_sensing | 30 | 0 | 30 | 0 | 0 |
| false positives | 1 | 0 | 0 | 1 | 0 |

Row-normalised (share of each reference category):

| Reference (macro) | n | physical_electrical | telemetry_and_sensing | atmosphere | other |
|---|---:|---:|---:|---:|---:|
| physical_electrical | 33 | 90.9% | 0.6% | 8.5% | 0.0% |
| telemetry_and_sensing | 30 | 0.0% | 100.0% | 0.0% | 0.0% |
| false positives | 1 | 0.0% | 0.0% | 100.0% | 0.0% |

### qwen3.5-9b (mean over 5 trials)

| Reference (macro) | n | physical_electrical | telemetry_and_sensing | atmosphere | other |
|---|---:|---:|---:|---:|---:|
| physical_electrical | 33 | 20.60 | 8 | 4.40 | 0 |
| telemetry_and_sensing | 30 | 0 | 30 | 0 | 0 |
| false positives | 1 | 0 | 0 | 0 | 1 |

Row-normalised (share of each reference category):

| Reference (macro) | n | physical_electrical | telemetry_and_sensing | atmosphere | other |
|---|---:|---:|---:|---:|---:|
| physical_electrical | 33 | 62.4% | 24.2% | 13.3% | 0.0% |
| telemetry_and_sensing | 30 | 0.0% | 100.0% | 0.0% | 0.0% |
| false positives | 1 | 0.0% | 0.0% | 0.0% | 100.0% |

## Correct predictions (mean over trials)

Number of reference episodes assigned to the **correct** macro category, averaged over the trials (share of the category in brackets).

| Model | physical_electrical (n=33) | telemetry_and_sensing (n=30) |
|---|---:|---:|
| gemma-4-e4b | 30 (90.9%) | 30 (100.0%) |
| qwen3.5-9b | 20.60 (62.4%) | 30 (100.0%) |

