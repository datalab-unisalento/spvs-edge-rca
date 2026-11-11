#!/usr/bin/env python3
"""Repeat the SPVS root-cause LLM ablation study N times per configuration and
aggregate the validation metrics as mean +/- standard deviation (sample std).

Reuses the parsing / inference / validation helpers from run_ablation.py.

A "trial" is a full pass over the selected configurations (64 events each).
Trial 1 reuses the work already done:
  - "full"      -> the original LM Studio log is copied verbatim
  - ablated     -> the existing single-run log (02_lms_log_stream_<config>.log)
                   is copied into the per-trial location
Trials 2..N run fresh LLM inference for every configuration (including "full").

If a per-trial log already exists it is reused, so the run is resumable after
an interruption (just re-invoke the same command).

Output layout (under --out-dir, default ablation_study/)
    logs/<config>/run_<k>/02_lms_log_stream.log   per-trial LM Studio log
    contexts/context_<config>.json                ablated context (shared by trials)
    validation/<config>/run_<k>/summary.json      per-trial validation output
    comparison.md / comparison.csv                aggregated mean +/- std per config

Examples:
  python ablation_study/run_ablation_repeats.py --repeats 5
  python ablation_study/run_ablation_repeats.py --repeats 5 --configs no_hints,minimal
  python ablation_study/run_ablation_repeats.py --repeats 2 --limit 2   # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import run_ablation as ra

CONFIGS = ra.CONFIGS
CONFIG_DESCRIPTIONS = ra.CONFIG_DESCRIPTIONS

# Metric name -> extractor that pulls a float from a trial validation summary.
METRIC_EXTRACTORS: Dict[str, Callable[[dict], Optional[float]]] = {
    "llm_fine_accuracy": lambda s: (s.get("ground_truth_vs_llm_metrics") or {}).get("accuracy"),
    "llm_fine_macro_f1": lambda s: (s.get("ground_truth_vs_llm_metrics") or {}).get("macro_f1"),
    "llm_fine_balanced_accuracy": lambda s: (s.get("ground_truth_vs_llm_metrics") or {}).get("balanced_accuracy"),
    "llm_fine_kappa": lambda s: (s.get("ground_truth_vs_llm_metrics") or {}).get("kappa"),
    "llm_macro_accuracy": lambda s: (s.get("ground_truth_vs_llm_macro") or {}).get("accuracy"),
    "llm_macro_macro_f1": lambda s: (s.get("ground_truth_vs_llm_macro") or {}).get("macro_f1"),
    "baseline_majority_fine_accuracy": lambda s: (s.get("ground_truth_vs_majority_metrics") or {}).get("accuracy"),
    "baseline_majority_macro_accuracy": lambda s: (s.get("ground_truth_vs_majority_macro") or {}).get("accuracy"),
    "baseline_dtree_fine_accuracy": lambda s: (s.get("ground_truth_vs_dtree_metrics") or {}).get("accuracy"),
    "baseline_dtree_macro_accuracy": lambda s: (s.get("ground_truth_vs_dtree_macro") or {}).get("accuracy"),
    "baseline_dtree_fine_macro_f1": lambda s: (s.get("ground_truth_vs_dtree_metrics") or {}).get("macro_f1"),
    "baseline_dtree_macro_macro_f1": lambda s: (s.get("ground_truth_vs_dtree_macro") or {}).get("macro_f1"),
    "avg_prompt_tokens": lambda s: (s.get("prompt_length") or {}).get("avg_prompt_tokens"),
    "median_prompt_tokens": lambda s: (s.get("prompt_length") or {}).get("median_prompt_tokens"),
    "avg_prompt_chars": lambda s: (s.get("prompt_length") or {}).get("avg_prompt_chars"),
    "failed_calls": lambda s: s.get("failed_calls"),
}

PERCENT_METRICS = {
    "llm_fine_accuracy",
    "llm_fine_macro_f1",
    "llm_fine_balanced_accuracy",
    "llm_macro_accuracy",
    "llm_macro_macro_f1",
    "baseline_majority_fine_accuracy",
    "baseline_majority_macro_accuracy",
    "baseline_dtree_fine_accuracy",
    "baseline_dtree_macro_accuracy",
    "baseline_dtree_fine_macro_f1",
    "baseline_dtree_macro_macro_f1",
}
KAPPA_METRICS = {"llm_fine_kappa"}
INT_METRICS = {"avg_prompt_tokens", "median_prompt_tokens", "avg_prompt_chars"}
COUNT_METRICS = {"failed_calls"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the root-cause LLM ablation study N times per config and aggregate mean +/- std."
    )
    parser.add_argument("--llm-log", type=Path, default=ra.DEFAULT_LLM_LOG)
    parser.add_argument("--detector-log", type=Path, default=ra.DEFAULT_DETECTOR_LOG)
    parser.add_argument("--classifier-log", type=Path, default=ra.DEFAULT_CLASSIFIER_LOG)
    parser.add_argument("--macro-config", type=Path, default=ra.DEFAULT_MACRO_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=ra.ROOT_DIR / "ablation_study",
                        help="Output directory (default: %(default)s).")
    parser.add_argument("--configs", default=",".join(CONFIGS),
                        help="Comma-separated configs (default: all).")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of trials per configuration (default: %(default)s).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Restrict to the first N events per trial (smoke test).")
    parser.add_argument("--api-base", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--reasoning-effort", default=None,
                        help="Pass 'reasoning_effort' to the API (e.g. 'none' disables thinking on reasoning models).")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force-infer", action="store_true",
                        help="Re-infer trials even if their logs already exist.")
    parser.add_argument("--no-reuse-trial1", action="store_true",
                        help="Infer trial 1 fresh for every config instead of reusing the original/legacy log (use when switching models).")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Do not call the LLM; reuse existing trial logs (missing trials get 0%% coverage).")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation on existing trial logs (no inference).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Event / prompt plumbing (mirrors run_ablation.main)
# ---------------------------------------------------------------------------
def build_events(original_log: Path, limit: Optional[int], model_override: Optional[str] = None):
    _, input_records, output_records = ra.load_log_records(original_log)
    if limit:
        input_records = input_records[:limit]
        output_records = output_records[:limit]

    model_identifier = str(
        (input_records[0].get("data") or {}).get("modelIdentifier") or "google/gemma-4-e4b"
    )
    model_path = str((input_records[0].get("data") or {}).get("modelPath") or model_identifier)
    if model_override:
        model_identifier = model_override
        model_path = model_override

    system_content: Optional[str] = None
    events: List[dict] = []
    for input_record, _output_record in zip(input_records, output_records):
        input_text: str = (input_record.get("data") or {}).get("input") or ""
        parsed = ra.extract_context(input_text)
        if parsed is None:
            raise SystemExit("Could not parse 'Context JSON:' from an input record.")
        context, _json_start, _json_end = parsed
        sys_content, user_content = ra.extract_messages(input_text)
        if system_content is None:
            system_content = sys_content
        elif sys_content != system_content:
            raise SystemExit("System prompt differs across input records; aborting.")
        events.append(
            {
                "input_timestamp": input_record.get("timestamp"),
                "context": context,
                "user_content": user_content,
            }
        )
    return events, system_content, model_identifier, model_path


def infer_config_log(
    config_log: Path,
    events: List[dict],
    system_content: str,
    transform: Callable[[dict], dict],
    model_path: str,
    model_identifier: str,
    args: argparse.Namespace,
) -> int:
    """Run fresh LLM inference for every event and write input+output records."""
    new_records: List[dict] = []
    ablated = [transform(ev["context"]) for ev in events]
    n_events = len(events)
    failures = 0
    for i, ev in enumerate(events, start=1):
        new_user = ra.replace_context(ev["user_content"], ablated[i - 1])
        new_input = ra.rebuild_input_text(system_content, new_user)
        new_records.append(ra.build_input_record(ev["input_timestamp"], new_input, model_path, model_identifier))

        content: Optional[str] = None
        stats: Optional[dict] = None
        error: Optional[str] = None
        for attempt in range(args.retries + 1):
            content, stats, error = ra.call_llm(
                system_content,
                new_user,
                api_base=args.api_base,
                model=model_identifier,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
            )
            if content is not None:
                break
            if error and error.startswith("HTTP 4") and attempt < args.retries:
                continue
            if error and not error.startswith("HTTP 4"):
                break
        if content is None:
            failures += 1
            content = ra.build_fallback_output(error or "unknown error")
            stats = {
                "totalTimeSec": 0.0,
                "promptTokensCount": None,
                "predictedTokensCount": None,
                "totalTokensCount": None,
            }
        new_records.append(ra.build_output_record(int(time.time() * 1000) + i, content, stats, model_identifier))
        if i % 10 == 0 or i == n_events:
            print(f"      event {i}/{n_events} done")
    ra.write_log(config_log, new_records)
    return failures


def trial_paths(config: str, k: int, repeats: int, logs_dir: Path, validation_dir: Path):
    if repeats == 1:
        return logs_dir / f"02_lms_log_stream_{config}.log", validation_dir / config
    return (
        logs_dir / config / f"run_{k}" / "02_lms_log_stream.log",
        validation_dir / config / f"run_{k}",
    )


def ensure_config_log(
    config_log: Path,
    config: str,
    k: int,
    events: List[dict],
    system_content: str,
    model_path: str,
    model_identifier: str,
    args: argparse.Namespace,
    original_log: Path,
    legacy_log: Optional[Path],
) -> str:
    """Return a status string after ensuring config_log exists."""
    config_log.parent.mkdir(parents=True, exist_ok=True)
    if config_log.exists() and not args.force_infer:
        return "reused"
    if config == "full" and k == 1 and args.limit is None and not args.no_reuse_trial1:
        shutil.copyfile(original_log, config_log)
        return "copied-original"
    if (
        k == 1
        and legacy_log is not None
        and legacy_log.exists()
        and args.limit is None
        and not args.force_infer
        and not args.no_reuse_trial1
    ):
        shutil.copyfile(legacy_log, config_log)
        return "copied-legacy"
    if args.skip_inference or args.validate_only:
        new_records = []
        for ev in events:
            new_user = ra.replace_context(ev["user_content"], ra.TRANSFORMS[config](ev["context"]))
            new_records.append(ra.build_input_record(
                ev["input_timestamp"], ra.rebuild_input_text(system_content, new_user), model_path, model_identifier
            ))
        ra.write_log(config_log, new_records)
        return "inputs-only"
    failures = infer_config_log(
        config_log, events, system_content, ra.TRANSFORMS[config], model_path, model_identifier, args
    )
    return f"inferred ({failures} failures)" if failures else "inferred"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _agg(values: List[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "values": []}
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": m, "std": sd, "values": list(values)}


def aggregate_config(config: str, trial_summaries: List[Tuple[int, str, dict]]) -> dict:
    metrics: Dict[str, dict] = {}
    for name, extractor in METRIC_EXTRACTORS.items():
        values = [extractor(s) for (_k, _status, s) in trial_summaries if extractor(s) is not None]
        metrics[name] = _agg(values)
    trial_info = []
    for k, status, s in trial_summaries:
        llm = s.get("ground_truth_vs_llm_metrics") or {}
        llm_macro = s.get("ground_truth_vs_llm_macro") or {}
        trial_info.append(
            {
                "trial": k,
                "status": status,
                "llm_fine_accuracy": llm.get("accuracy"),
                "llm_fine_macro_f1": llm.get("macro_f1"),
                "llm_fine_kappa": llm.get("kappa"),
                "llm_macro_accuracy": llm_macro.get("accuracy"),
                "llm_macro_macro_f1": llm_macro.get("macro_f1"),
                "llm_macro_macro_precision": llm_macro.get("macro_precision"),
                "llm_macro_macro_recall": llm_macro.get("macro_recall"),
                "llm_macro_balanced_accuracy": llm_macro.get("balanced_accuracy"),
                "failed_calls": s.get("failed_calls", 0),
            }
        )
    return {
        "config": config,
        "description": CONFIG_DESCRIPTIONS[config],
        "trials": len(trial_summaries),
        "metrics": metrics,
        "trial_info": trial_info,
    }


# ---------------------------------------------------------------------------
# Comparison report (aggregated mean +/- std)
# ---------------------------------------------------------------------------
def _fmt(metric: str, agg: dict) -> str:
    m, sd = agg.get("mean"), agg.get("std")
    if m is None:
        return "n/a"
    if metric in PERCENT_METRICS:
        return f"{m * 100.0:.2f}%±{sd * 100.0:.2f}%"
    if metric in KAPPA_METRICS:
        return f"{m:.3f}±{sd:.3f}"
    if metric in COUNT_METRICS:
        return f"{m:.1f}±{sd:.1f}"
    return f"{m:.0f}±{sd:.0f}"


def _csv_rows(summaries: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for config in CONFIGS:
        agg = summaries.get(config)
        if agg is None:
            continue
        row = {
            "config": config,
            "description": agg["description"],
            "trials": agg["trials"],
        }
        for name in METRIC_EXTRACTORS:
            metrics = agg["metrics"].get(name) or {}
            row[f"{name}_mean"] = "" if metrics.get("mean") is None else f"{metrics['mean']:.6f}"
            row[f"{name}_std"] = "" if metrics.get("std") is None else f"{metrics['std']:.6f}"
            row[name] = _fmt(name, metrics)
        rows.append(row)
    return rows


def write_comparison(out_dir: Path, summaries: Dict[str, dict]) -> None:
    rows = _csv_rows(summaries)
    if rows:
        import csv as _csv

        fieldnames = list(rows[0].keys())
        with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    lines = ["# Root-Cause LLM Ablation Study (mean ± std over repeated trials)", ""]
    lines.append("Each configuration is evaluated on repeated independent LLM inference passes.")
    lines.append("Metrics are the mean over trials with the sample standard deviation as the error.")
    lines.append("Ground truth and baseline are computed from the **full** context (fixed reference).")
    lines.append("")
    lines.append(
        "| Config | Description | Trials | Fine acc. | Fine macro F1 | Fine bal. acc. | Fine kappa "
        "| Macro acc. | Macro F1 | Baseline maj. fine | Baseline maj. macro | Baseline DT fine | Baseline DT macro "
        "| Avg prompt tok | Med prompt tok | Avg chars | Failed calls |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['description']} | {row['trials']} "
            f"| {row['llm_fine_accuracy']} | {row['llm_fine_macro_f1']} | {row['llm_fine_balanced_accuracy']} "
            f"| {row['llm_fine_kappa']} | {row['llm_macro_accuracy']} | {row['llm_macro_macro_f1']} "
            f"| {row['baseline_majority_fine_accuracy']} | {row['baseline_majority_macro_accuracy']} "
            f"| {row['baseline_dtree_fine_accuracy']} | {row['baseline_dtree_macro_accuracy']} "
            f"| {row['avg_prompt_tokens']} | {row['median_prompt_tokens']} "
            f"| {row['avg_prompt_chars']} | {row['failed_calls']} |"
        )
    lines.append("")
    lines.append("## Details per configuration")
    lines.append("")
    for config in CONFIGS:
        agg = summaries.get(config)
        if agg is None:
            continue
        lines.append(f"### {config} — {agg['description']}")
        lines.append("")
        lines.append("| Trial | Status | Fine acc. | Fine macro F1 | Fine kappa | Macro acc. | Macro F1 | Macro precision | Macro recall | Macro bal. acc. | Failed calls |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for info in agg["trial_info"]:
            pct = lambda v: "n/a" if v is None else f"{v * 100:.2f}%"
            num = lambda v: "n/a" if v is None else f"{v:.3f}"
            failed = int(info.get("failed_calls") or 0)
            lines.append(
                f"| {info['trial']} | {info['status']} "
                f"| {pct(info.get('llm_fine_accuracy'))} | {pct(info.get('llm_fine_macro_f1'))} | {num(info.get('llm_fine_kappa'))} "
                f"| {pct(info.get('llm_macro_accuracy'))} | {pct(info.get('llm_macro_macro_f1'))} "
                f"| {pct(info.get('llm_macro_macro_precision'))} | {pct(info.get('llm_macro_macro_recall'))} "
                f"| {pct(info.get('llm_macro_balanced_accuracy'))} | {failed} |"
            )
        lines.append("")
        lines.append(f"- Trials: {agg['trials']}")
        for name in ("llm_fine_accuracy", "llm_fine_macro_f1", "llm_fine_balanced_accuracy", "llm_fine_kappa",
                     "llm_macro_accuracy", "llm_macro_macro_f1",
                     "baseline_majority_fine_accuracy", "baseline_majority_macro_accuracy",
                     "baseline_dtree_fine_accuracy", "baseline_dtree_macro_accuracy",
                     "baseline_dtree_fine_macro_f1", "baseline_dtree_macro_macro_f1",
                     "avg_prompt_tokens", "median_prompt_tokens", "avg_prompt_chars",
                     "failed_calls"):
            lines.append(f"- {name}: {_fmt(name, agg['metrics'].get(name) or {})}")
        total_failed = sum(int(info.get("failed_calls") or 0) for info in agg["trial_info"])
        lines.append(f"- Total failed calls (sum across trials): {total_failed}")
        lines.append("")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "comparison.csv").write_text("", encoding="utf-8") if not rows else None
    print(f"Aggregated comparison written to {out_dir / 'comparison.md'}")
    if rows:
        print(f"Aggregated comparison CSV written to {out_dir / 'comparison.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    original_log: Path = args.llm_log.resolve()
    if not original_log.exists():
        raise SystemExit(f"Original log not found: {original_log}")
    detector_log = args.detector_log.resolve()
    classifier_log = args.classifier_log.resolve()
    macro_config = args.macro_config.resolve()
    out_dir = args.out_dir.resolve()
    logs_dir = out_dir / "logs"
    contexts_dir = out_dir / "contexts"
    validation_dir = out_dir / "validation"
    logs_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}. Valid: {', '.join(CONFIGS)}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    events, system_content, model_identifier, model_path = build_events(original_log, args.limit, args.model)
    print(f"Parsed {len(events)} events. Model: {model_identifier}. Repeats per config: {args.repeats}")

    summaries: Dict[str, dict] = {}
    for config in configs:
        print(f"\n=== Config: {config} ({CONFIG_DESCRIPTIONS[config]}) ===")
        ra.write_contexts(contexts_dir / f"context_{config}.json", [ra.TRANSFORMS[config](ev["context"]) for ev in events])

        legacy_log = logs_dir / f"02_lms_log_stream_{config}.log"
        trial_summaries: List[Tuple[int, str, dict]] = []
        for k in range(1, args.repeats + 1):
            config_log, validation_out = trial_paths(config, k, args.repeats, logs_dir, validation_dir)
            status = ensure_config_log(
                config_log, config, k, events, system_content, model_path, model_identifier,
                args, original_log, legacy_log,
            )
            print(f"  trial {k}/{args.repeats}: {config_log} [{status}]")
            if args.validate_only and not config_log.exists():
                print("    -> trial log missing, skipping validation for this trial.")
                continue
            summary = ra.run_validation(
                config_log=config_log,
                ground_truth_log=original_log,
                out_dir=validation_out,
                detector_log=detector_log,
                classifier_log=classifier_log,
                macro_config=macro_config,
            )
            summary["prompt_length"] = ra.compute_prompt_length_stats(config_log)
            summary["failed_calls"] = ra.count_failed_calls(config_log)
            llm = summary.get("ground_truth_vs_llm_metrics") or {}
            print(f"    -> fine acc {llm.get('accuracy', 0.0) * 100:.2f}%, "
                  f"macro F1 {llm.get('macro_f1', 0.0) * 100:.2f}%")
            trial_summaries.append((k, status, summary))

        summaries[config] = aggregate_config(config, trial_summaries)
        m = summaries[config]["metrics"]
        print(
            f"  aggregated fine acc = {_fmt('llm_fine_accuracy', m['llm_fine_accuracy'])}, "
            f"macro F1 = {_fmt('llm_fine_macro_f1', m['llm_fine_macro_f1'])}"
        )

    write_comparison(out_dir, summaries)
    print("\nRepeated ablation study complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
