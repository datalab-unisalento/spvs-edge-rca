# Additional inputs for report reproduction

This folder contains the report inputs not already available in cameraready/data/. All 62 files are hash-verified in MANIFEST.json and have been checked for duplicate files in data/.

- ablation_study/rerun_20260821/: 25 Gemma trial summaries and their 25 replay logs; five Qwen full-context summaries and their five replay logs under qwen3.5-9b/.
- ablation_study/rerun_20260821/lms_stream_gemma.jsonl: matching native Gemma runtime capture, 320 full-context events.
- ablation_study/rerun_20260821/qwen3.5-9b/lms_stream_qwen.jsonl: matching native Qwen runtime capture, 320 full-context events.

The native JSONL streams supply the original generation throughput, time-to-first-token and total-time statistics. Replay logs contain client-observed API duration instead and are not substituted for native timings. The unrelated logsLLM.txt and logsLLMQWEN.txt captures are not needed and are no longer included.

Original directory names are preserved within this package; they are not external dependencies. Absolute paths recorded inside summary metadata are inert and are not used to locate inputs. Latency extraction reuses ../../data/logs/ without copying its data here.

See ../README.md for commands. No model server or network access is required for report aggregation.
