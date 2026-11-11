#!/usr/bin/env python3
"""Compare two LLMs on the `full` configuration of the root-cause ablation study.

Reads the per-trial validation summaries and trial logs of the `full` config from
two model directories and reports each metric as mean +/- std over the trials.

Reuses the parsing / aggregation helpers from run_ablation.py and
run_ablation_repeats.py.

Example:
  python ablation_study/compare_models.py \
      --model-a-dir ablation_study \
      --model-b-dir ablation_study/model_compare/qwen3.5-9b \
      --model-a-label gemma-4-e4b --model-b-label qwen3.5-9b-q4_k_m \
      --out ablation_study/model_compare/model_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import run_ablation as ra
import run_ablation_repeats as rr

CONFIG = "full"
METRIC_ORDER = [
    "llm_fine_accuracy",
    "llm_fine_macro_f1",
    "llm_fine_balanced_accuracy",
    "llm_fine_kappa",
    "llm_macro_accuracy",
    "llm_macro_macro_f1",
    "baseline_majority_fine_accuracy",
    "baseline_majority_macro_accuracy",
    "baseline_dtree_fine_accuracy",
    "baseline_dtree_macro_accuracy",
    "baseline_dtree_fine_macro_f1",
    "baseline_dtree_macro_macro_f1",
    "baseline_dtree_fine_kappa",
    "avg_prompt_tokens",
    "median_prompt_tokens",
    "avg_prompt_chars",
    "failed_calls",
]

METRIC_LABELS = {
    "llm_fine_accuracy": "Fine accuracy",
    "llm_fine_macro_f1": "Fine macro F1",
    "llm_fine_balanced_accuracy": "Fine balanced accuracy",
    "llm_fine_kappa": "Fine kappa",
    "llm_macro_accuracy": "Macro accuracy",
    "llm_macro_macro_f1": "Macro F1",
    "baseline_majority_fine_accuracy": "Baseline majority fine accuracy",
    "baseline_majority_macro_accuracy": "Baseline majority macro accuracy",
    "baseline_dtree_fine_accuracy": "Baseline decision tree fine accuracy",
    "baseline_dtree_macro_accuracy": "Baseline decision tree macro accuracy",
    "baseline_dtree_fine_macro_f1": "Baseline decision tree fine macro F1",
    "baseline_dtree_macro_macro_f1": "Baseline decision tree macro macro F1",
    "baseline_dtree_fine_kappa": "Baseline decision tree fine kappa",
    "avg_prompt_tokens": "Avg prompt tokens",
    "median_prompt_tokens": "Median prompt tokens",
    "avg_prompt_chars": "Avg prompt chars",
    "failed_calls": "Failed calls",
}

# Extra metrics on top of the shared extractors in run_ablation_repeats.
EXTRA_METRIC_EXTRACTORS: Dict[str, Callable[[dict], Optional[float]]] = {
    "baseline_dtree_fine_kappa": lambda s: (s.get("ground_truth_vs_dtree_metrics") or {}).get("kappa"),
}
METRIC_EXTRACTORS: Dict[str, Callable[[dict], Optional[float]]] = {
    **rr.METRIC_EXTRACTORS,
    **EXTRA_METRIC_EXTRACTORS,
}
# Kappa metrics not known to run_ablation_repeats._fmt (which would round them as integers).
EXTRA_KAPPA_METRICS = {"baseline_dtree_fine_kappa"}

# Performance/latency stats extracted from the raw LM Studio log streams.
STATS = [
    ("prompt_tokens", "Prompt tokens"),
    ("predicted_tokens", "Predicted tokens"),
    ("total_tokens", "Total tokens"),
    ("tokens_per_second", "Tokens per second"),
    ("time_to_first_token_sec", "Time to first token (s)"),
    ("total_time_sec", "Total time (s)"),
    ("confidence", "Confidence"),
    ("failed_calls", "Failed calls"),
]


def _stat_fmt(agg: dict) -> str:
    m, sd = agg.get("mean"), agg.get("std")
    if m is None:
        return "n/a"
    return f"{m:.2f}±{sd:.2f}"


def load_model_trials(model_dir: Path) -> List[dict]:
    """Return the per-trial summary dicts (with prompt_length/failed_calls)."""
    val_dir = model_dir / "validation" / CONFIG
    logs_dir = model_dir / "logs" / CONFIG
    if not val_dir.is_dir():
        return []
    trials: List[dict] = []
    for run_dir in sorted(val_dir.glob("run_*"), key=lambda p: int(p.name.split("_")[1])):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        k = int(run_dir.name.split("_")[1])
        trial_log = logs_dir / f"run_{k}" / "02_lms_log_stream.log"
        summary["prompt_length"] = ra.compute_prompt_length_stats(trial_log)
        summary["failed_calls"] = ra.count_failed_calls(trial_log)
        trials.append(summary)
    return trials


def aggregate(trials: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, extractor in METRIC_EXTRACTORS.items():
        values = [extractor(s) for s in trials if extractor(s) is not None]
        out[name] = rr._agg(values)
    return out


def _fmt(metric: str, agg: dict) -> str:
    if metric in EXTRA_KAPPA_METRICS:
        m, sd = agg.get("mean"), agg.get("std")
        if m is None:
            return "n/a"
        return f"{m:.3f}±{sd:.3f}"
    return rr._fmt(metric, agg)


# ---------------------------------------------------------------------------
# LM Studio stats-log parsing (raw `lms log stream --json --stats` captures)
# ---------------------------------------------------------------------------
def read_records(path: Path) -> List[dict]:
    raw = path.read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-16", "utf-8", "utf-16-le", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        raise SystemExit(f"cannot decode {path}")
    records: List[dict] = []
    for line in text.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def extract_context(input_text: str):
    marker = "Context JSON:"
    idx = input_text.find(marker)
    if idx < 0:
        return None
    js = input_text.find("{", idx)
    if js < 0:
        return None
    dec = json.JSONDecoder()
    try:
        ctx, _ = dec.raw_decode(input_text[js:])
    except json.JSONDecodeError:
        return None
    return ctx if isinstance(ctx, dict) else None


def classify_config(context) -> str:
    if context is None:
        return "unknown"
    if set(context.keys()) == {"event", "detection_context"}:
        return "minimal"
    history = context.get("history") or {}
    if "inference_hints" not in context:
        return "no_hints"
    if "weather" not in history or "weather_latest" not in history:
        return "no_weather"
    if "baseline_same_hour" not in history:
        return "no_baseline"
    return "full"


def parse_stats_log(path: Path, target_config: str = "full") -> List[dict]:
    """Return per-event stats (tokens/latency/confidence/failures) for target_config."""
    recs = read_records(path)
    inputs = [r for r in recs if (r.get("data") or {}).get("type") == "llm.prediction.input"]
    outputs = [r for r in recs if (r.get("data") or {}).get("type") == "llm.prediction.output"]
    events: List[dict] = []
    for inp, out in zip(inputs, outputs):
        ctx = extract_context((inp.get("data") or {}).get("input") or "")
        if classify_config(ctx) != target_config:
            continue
        od = out.get("data") or {}
        stats = od.get("stats") or {}
        out_str = od.get("output") or ""
        try:
            confidence = json.loads(out_str).get("confidence")
        except (json.JSONDecodeError, AttributeError, TypeError):
            confidence = None
        failed = 1 if ("Root cause unavailable" in out_str or "llm_error" in out_str) else 0
        events.append(
            {
                "prompt_tokens": stats.get("promptTokensCount"),
                "predicted_tokens": stats.get("predictedTokensCount"),
                "total_tokens": stats.get("totalTokensCount"),
                "tokens_per_second": stats.get("tokensPerSecond"),
                "time_to_first_token_sec": stats.get("timeToFirstTokenSec"),
                "total_time_sec": stats.get("totalTimeSec"),
                "confidence": confidence,
                "failed_calls": failed,
            }
        )
    return events


def aggregate_stats(events: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for key, _label in STATS:
        values = [e[key] for e in events if e[key] is not None]
        out[key] = rr._agg(values)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two LLMs on the full-config ablation runs.")
    p.add_argument("--model-a-dir", type=Path, required=True, help="Dir with model A's logs/validation (full).")
    p.add_argument("--model-b-dir", type=Path, required=True, help="Dir with model B's logs/validation (full).")
    p.add_argument("--model-a-label", default="model_a")
    p.add_argument("--model-b-label", default="model_b")
    p.add_argument("--model-a-stats-log", type=Path, default=None,
                    help="Raw LM Studio log stream for model A (stats extracted for full config).")
    p.add_argument("--model-b-stats-log", type=Path, default=None,
                    help="Raw LM Studio log stream for model B (stats extracted for full config).")
    p.add_argument("--stats-from-trials", action="store_true",
                   help="Use the same full-context trial logs as the classification scores for performance statistics.")
    p.add_argument("--out", type=Path, default=Path("ablation_study/model_compare/model_comparison"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    trials_a = load_model_trials(args.model_a_dir.resolve())
    trials_b = load_model_trials(args.model_b_dir.resolve())
    if not trials_a or not trials_b:
        raise SystemExit(
            f"Need trial summaries for 'full' in both dirs (got A={len(trials_a)}, B={len(trials_b)})."
        )
    agg_a = aggregate(trials_a)
    agg_b = aggregate(trials_b)

    stats_a = stats_b = None
    n_stats_a = n_stats_b = 0
    if args.stats_from_trials:
        if args.model_a_stats_log or args.model_b_stats_log:
            raise SystemExit('Use --stats-from-trials or explicit raw stats logs, not both.')
        events = []
        for directory in [args.model_a_dir, args.model_b_dir]:
            combined = []
            for trial in sorted((directory/'validation/full').glob('run_*')):
                log = directory/'logs/full'/trial.name/'02_lms_log_stream.log'
                parsed = parse_stats_log(log)
                if len(parsed) != 64:
                    raise SystemExit(f'Expected 64 full-context events in {log}; got {len(parsed)}')
                combined.extend(parsed)
            if len(combined) != 320:
                raise SystemExit(f'Expected 320 events across five trials in {directory}; got {len(combined)}')
            events.append(combined)
        stats_a, stats_b = [aggregate_stats(e) for e in events]
        n_stats_a, n_stats_b = [len(e) for e in events]
    elif args.model_a_stats_log:
        recs_a = parse_stats_log(args.model_a_stats_log.resolve())
        stats_a = aggregate_stats(recs_a)
        n_stats_a = len(recs_a)
    if args.model_b_stats_log:
        recs_b = parse_stats_log(args.model_b_stats_log.resolve())
        stats_b = aggregate_stats(recs_b)
        n_stats_b = len(recs_b)

    lines = ["# LLM Comparison — full configuration (mean ± std over trials)", ""]
    lines.append(f"- Model A: **{args.model_a_label}** ({len(trials_a)} trials)")
    lines.append(f"- Model B: **{args.model_b_label}** ({len(trials_b)} trials)")
    lines.append("- Ground truth and baseline are computed from the **full** context (fixed reference).")
    lines.append("")
    lines.append("| Metric | Model A | Model B |")
    lines.append("|---|---|---|")
    for name in METRIC_ORDER:
        fmt_a = _fmt(name, agg_a.get(name) or {})
        fmt_b = _fmt(name, agg_b.get(name) or {})
        lines.append(f"| {METRIC_LABELS.get(name, name)} | {fmt_a} | {fmt_b} |")

    for label, trials, agg in ((args.model_a_label, trials_a, agg_a), (args.model_b_label, trials_b, agg_b)):
        lines.append("")
        lines.append(f"### {label} — per trial")
        lines.append("")
        lines.append("| Trial | Fine acc. | Fine macro F1 | Fine kappa | Macro acc. | Macro F1 | Macro precision | Macro recall | Macro bal. acc. | Failed calls |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in trials:
            llm = s.get("ground_truth_vs_llm_metrics") or {}
            llm_macro = s.get("ground_truth_vs_llm_macro") or {}
            pct = lambda v: "n/a" if v is None else f"{v * 100:.2f}%"
            num = lambda v: "n/a" if v is None else f"{v:.3f}"
            failed = int(s.get("failed_calls") or 0)
            lines.append(
                f"| {trials.index(s) + 1} "
                f"| {pct(llm.get('accuracy'))} | {pct(llm.get('macro_f1'))} | {num(llm.get('kappa'))} "
                f"| {pct(llm_macro.get('accuracy'))} | {pct(llm_macro.get('macro_f1'))} "
                f"| {pct(llm_macro.get('macro_precision'))} | {pct(llm_macro.get('macro_recall'))} "
                f"| {pct(llm_macro.get('balanced_accuracy'))} | {failed} |"
            )

    if stats_a is not None and stats_b is not None:
        lines.append("")
        lines.append("## Inference performance (from LM Studio logs — full config)")
        lines.append("")
        lines.append(f"- Full-config events: {args.model_a_label}={n_stats_a}, {args.model_b_label}={n_stats_b}")
        lines.append("")
        lines.append("| Stat | Model A | Model B |")
        lines.append("|---|---|---|")
        for key, label in STATS:
            lines.append(f"| {label} | {_stat_fmt(stats_a.get(key) or {})} | {_stat_fmt(stats_b.get(key) or {})} |")
        if args.stats_from_trials:
            lines.extend([
                "",
                "Statistics use the same five archived trial logs per model as the classification tables (320 events each).",
                "In these replay logs, total time is the client-observed API round-trip duration, and tokens per second is completion tokens divided by that duration.",
                "These are not native LM Studio generation-only timing measurements. Time to first token was not recorded and is reported as n/a.",
                "This section is a reproducible replacement derived from the selected trial logs; it does not claim to recover the historical report's native runtime measurements.",
            ])

    out_md = Path(str(args.out) + ".md")
    out_csv = Path(str(args.out) + ".csv")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # CSV: rows = metrics, columns = A_mean/A_std/B_mean/B_std
    import csv as _csv

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.writer(handle)
        writer.writerow(["metric", f"{args.model_a_label}_mean", f"{args.model_a_label}_std",
                         f"{args.model_b_label}_mean", f"{args.model_b_label}_std"])
        for name in METRIC_ORDER:
            a = agg_a.get(name) or {}
            b = agg_b.get(name) or {}
            writer.writerow([
                name,
                "" if a.get("mean") is None else f"{a['mean']:.6f}",
                "" if a.get("std") is None else f"{a['std']:.6f}",
                "" if b.get("mean") is None else f"{b['mean']:.6f}",
                "" if b.get("std") is None else f"{b['std']:.6f}",
            ])
        if stats_a is not None and stats_b is not None:
            for key, _label in STATS:
                a = stats_a.get(key) or {}
                b = stats_b.get(key) or {}
                writer.writerow([
                    f"stat_{key}",
                    "" if a.get("mean") is None else f"{a['mean']:.6f}",
                    "" if a.get("std") is None else f"{a['std']:.6f}",
                    "" if b.get("mean") is None else f"{b['mean']:.6f}",
                    "" if b.get("std") is None else f"{b['std']:.6f}",
                ])

    print(f"Model comparison written to {out_md}")
    print(f"Model comparison CSV written to {out_csv}")
    print()
    for name in METRIC_ORDER:
        print(f"{METRIC_LABELS.get(name, name):28s} A={_fmt(name, agg_a.get(name) or {})}   B={_fmt(name, agg_b.get(name) or {})}")
    if stats_a is not None and stats_b is not None:
        print()
        for key, label in STATS:
            print(f"{label:28s} A={_stat_fmt(stats_a.get(key) or {})}   B={_stat_fmt(stats_b.get(key) or {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
