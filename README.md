# Diagnosis at the Source: Edge-Local Root-Cause Analysis for Distributed Photovoltaic Generation

Supporting materials for the BigData 2026 poster paper:  **Diagnosis at the Source: Edge-Local Root-Cause Analysis for Distributed Photovoltaic Generation**.

| Folder | Contents |
|---|---|
| `paper/` | Paper and accompanying publication materials. |
| `prompt/` | System and user prompts, model metadata, and inference settings. |
| `rules/` | Inference-hint generation and evaluation labelling rules, with readable documentation. |
| `schema/` | JSON schemas for anomaly events and the context supplied to the LLM. |
| `emulator/` | Emulator source code and campaign configuration. |
| `data/` | Telemetry, campaign logs, episode contexts, and raw model responses. |
| `results/` | Evaluation reports, labels and predictions, latency measurements, and self-contained report reproduction inputs and scripts. |
| `deploy/` | Docker Compose files, service configurations, and environment-variable setup. |

See [results/README.md](results/README.md) for instructions to reproduce the reports and [deploy/README.md](deploy/README.md) for deployment instructions.
