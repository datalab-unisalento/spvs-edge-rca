## Reproduce the reports

Requires Python >= 3.10 and only its standard library. From root folder:

```sh
python -B results/scripts/reproduce_reports.py --check-inputs
python -B results/scripts/reproduce_reports.py
```

The first command checks all 62 required inputs and their hashes. The second regenerates the three reports and companion CSVs in results/reproduced/ and compares each Markdown report with the supplied original. No LLM requests, training, relabeling or new per-episode evaluation are performed. Original reports remain untouched.

The bundle includes 30 archived evaluation summaries, 30 replay logs and two native JSONL timing streams. The native streams are report_inputs/ablation_study/rerun_20260821/lms_stream_gemma.jsonl and report_inputs/ablation_study/rerun_20260821/qwen3.5-9b/lms_stream_qwen.jsonl. They supply 320 full-context events per model, including measured time to first token.

The helpers run_ablation.py and run_ablation_repeats.py are imported for parsing and aggregation; their inference/validation entry points are not executed by the wrapper. No numerical values are hard-coded to make the reports agree.

## Episode predictions

`predictions.csv` is exported from the 30 replay logs under
`report_inputs/ablation_study/rerun_20260821/`, including the five Qwen runs
under `qwen3.5-9b/`. Each response is paired with its input context in the
sequential replay log, then joined to `labels.csv` by `episode_key`
(anomaly type and UTC start time). Run from `cameraready/`:

```sh
python -B results/scripts/rebuild_predictions.py
python -B results/scripts/rebuild_predictions.py --check
```

The exporter checks unique keys, complete coverage, both fine-category and
macro confusion matrices and accuracies, and all archived mismatch rows
(including confidence and evidence) against the corresponding `summary.json`
before writing. `--check` also compares every CSV field with the replay export
without writing files. There are 1,920 rows: 64 episodes across 30 runs.


## Latency measurements

latency/latencies.csv contains existing episode measurements; latency/latency_stats.csv contains the mean, standard deviation, median, and 95th percentile (p95) for each phase across the 63 episodes.

The marker summary is a diagnostic, not a replacement for the episode table. Stage t7 (Alertmanager to Telegram) was not instrumented.

