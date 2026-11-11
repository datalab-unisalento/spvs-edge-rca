#!/usr/bin/env python3
"""Macro-category confusion matrix for the full-configuration ablation runs.

Rows
    physical_electrical          reference macro = physical_electrical
    telemetry_and_sensing        reference macro = telemetry_and_sensing
    false positives              reference macro = atmospheric_effects
                                 (weather/environmental effect = detector false
                                 alarm from an O&M standpoint)

Columns
    n                            number of reference macro labels in that row
    physical_electrical          predicted macro = physical_electrical
    telemetry_and_sensing        predicted macro = telemetry_and_sensing
    atmosphere                   predicted macro = atmospheric_effects
    other                        every other predicted macro label
                                 (unknown, degradation, contamination, ...)

Every predicted cell is the *mean count over the N trials*, so the printed values
can be fractional (e.g. 2.80 = the model predicted this bucket 2.8 times per run
on average).

Examples
    python ablation_study/macro_confusion_matrix.py \
        --model gemma-4-e4b=ablation_study/rerun_20260821 \
        --model qwen3.5-9b=ablation_study/rerun_20260821/qwen3.5-9b \
        --repeats 5 \
        --out ablation_study/rerun_20260821/macro_confusion_matrix
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

# (display label, underlying macro label in summary.json)
ROW_ORDER: List[Tuple[str, str]] = [
    ("physical_electrical", "physical_electrical"),
    ("telemetry_and_sensing", "telemetry_and_sensing"),
    ("false positives", "atmospheric_effects"),
]

PRED_COLUMNS: List[Tuple[str, str]] = [
    ("physical_electrical", "physical_electrical"),
    ("telemetry_and_sensing", "telemetry_and_sensing"),
    ("atmosphere", "atmospheric_effects"),
]

OTHER = "other"
COLUMN_ORDER = ["n"] + [disp for disp, _src in PRED_COLUMNS] + [OTHER]

# Predicted column that counts as "correct" for each reference row.
CORRECT_COLUMN: Dict[str, str] = {
    "physical_electrical": "physical_electrical",
    "telemetry_and_sensing": "telemetry_and_sensing",
    "false positives": "atmosphere",
}

# Rows reported in the cross-model "correct predictions" summary.
REQUESTED_ROWS: List[Tuple[str, str]] = [
    ("physical_electrical", "physical_electrical"),
    ("telemetry_and_sensing", "telemetry_and_sensing"),
]


def _load_confusion_matrix(summary_path: Path) -> Dict[str, Dict[str, float]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    macro = summary.get("ground_truth_vs_llm_macro") or {}
    return macro.get("confusion_matrix") or {}


def _trial_counts(confusion: Dict[str, Dict[str, float]]):
    """Collapse one trial's confusion matrix onto the requested rows/columns."""
    counts: Dict[str, Dict[str, float]] = {}
    reference_n: Dict[str, float] = {}
    for row_display, row_source in ROW_ORDER:
        predicted = confusion.get(row_source) or {}
        row = {column: 0.0 for column, _src in PRED_COLUMNS}
        row[OTHER] = 0.0
        total = 0.0
        for predicted_label, value in predicted.items():
            total += float(value)
            bucket = OTHER
            for column, source in PRED_COLUMNS:
                if predicted_label == source:
                    bucket = column
                    break
            row[bucket] += float(value)
        counts[row_display] = row
        reference_n[row_display] = total
    return counts, reference_n


def compute_model(model_dir: Path, repeats: int) -> List[dict]:
    """Average the confusion counts over the available trials of one model."""
    trials: List[Tuple[Dict[str, Dict[str, float]], Dict[str, float]]] = []
    used = 0
    for k in range(1, repeats + 1):
        summary_path = model_dir / "validation" / "full" / f"run_{k}" / "summary.json"
        if not summary_path.exists():
            continue
        counts, reference_n = _trial_counts(_load_confusion_matrix(summary_path))
        trials.append((counts, reference_n))
        used += 1
    if not trials:
        raise SystemExit(f"No trial summaries found under {model_dir / 'validation' / 'full'}")

    rows: List[dict] = []
    for row_display, row_source in ROW_ORDER:
        n_values = [reference_n[row_display] for _c, reference_n in trials]
        cells: Dict[str, float] = {}
        for column, _src in PRED_COLUMNS:
            values = [counts[row_display][column] for counts, _n in trials]
            cells[column] = sum(values) / len(values)
        other_values = [counts[row_display][OTHER] for counts, _n in trials]
        cells[OTHER] = sum(other_values) / len(other_values)
        rows.append(
            {
                "reference": row_display,
                "reference_source": row_source,
                "n": sum(n_values) / len(n_values),
                "cells": cells,
                "correct": cells[CORRECT_COLUMN[row_display]],
                "trials": used,
            }
        )
    return rows


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def _pct(part: float, total: float) -> str:
    if not total:
        return "n/a"
    return f"{part / total * 100.0:.1f}%"


def _model_block(label: str, rows: List[dict]) -> List[str]:
    trials = rows[0]["trials"] if rows else 0
    lines = [f"### {label} (mean over {trials} trials)", ""]
    lines.append("| Reference (macro) | n | physical_electrical | telemetry_and_sensing | atmosphere | other |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        cells = row["cells"]
        lines.append(
            f"| {row['reference']} | {_fmt(row['n'])} "
            f"| {_fmt(cells['physical_electrical'])} | {_fmt(cells['telemetry_and_sensing'])} "
            f"| {_fmt(cells['atmosphere'])} | {_fmt(cells[OTHER])} |"
        )
    lines.append("")
    lines.append("Row-normalised (share of each reference category):")
    lines.append("")
    lines.append("| Reference (macro) | n | physical_electrical | telemetry_and_sensing | atmosphere | other |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        cells = row["cells"]
        total = row["n"]
        lines.append(
            f"| {row['reference']} | {_fmt(total)} "
            f"| {_pct(cells['physical_electrical'], total)} | {_pct(cells['telemetry_and_sensing'], total)} "
            f"| {_pct(cells['atmosphere'], total)} | {_pct(cells[OTHER], total)} |"
        )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Macro-category confusion matrix for full-config runs.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="Model label and directory containing validation/full/run_<k>/summary.json (repeatable).",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Trials per model (default: 5).")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("ablation_study/rerun_20260821/macro_confusion_matrix"),
        help="Output basename (writes .md and .csv).",
    )
    args = parser.parse_args()

    models: List[Tuple[str, Path, List[dict]]] = []
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model must be LABEL=DIR, got: {spec}")
        label, directory = spec.split("=", 1)
        model_dir = Path(directory).resolve()
        rows = compute_model(model_dir, args.repeats)
        models.append((label, model_dir, rows))

    lines = ["# Macro-category confusion matrix (full configuration)", ""]
    lines.append("Rows are the **reference** macro labels, columns are the **predicted** macro labels.")
    lines.append("`false positives` collects the reference `atmospheric_effects` episodes (weather/environmental")
    lines.append("effects that are not real faults). `other` collects every other predicted macro label")
    lines.append("(e.g. `unknown`). Predicted cells are averaged over the trials, so they can be fractional.")
    lines.append("")
    for label, _directory, rows in models:
        lines.extend(_model_block(label, rows))

    # ---- Cross-model: how many episodes are correct on average ----
    lines.append("## Correct predictions (mean over trials)")
    lines.append("")
    lines.append(
        "Number of reference episodes assigned to the **correct** macro category, averaged over the trials "
        "(share of the category in brackets)."
    )
    lines.append("")
    header_cells: List[str] = []
    for row_display, _src in REQUESTED_ROWS:
        n_value = next((r["n"] for r in models[0][2] if r["reference"] == row_display), 0.0)
        header_cells.append(f"{row_display} (n={_fmt(n_value)})")
    lines.append("| Model | " + " | ".join(header_cells) + " |")
    lines.append("|---|" + "---:|" * len(header_cells))
    for label, _directory, rows in models:
        by_row = {row["reference"]: row for row in rows}
        cells: List[str] = []
        for row_display, _src in REQUESTED_ROWS:
            row = by_row.get(row_display)
            if row is None:
                cells.append("n/a")
            else:
                cells.append(f"{_fmt(row['correct'])} ({_pct(row['correct'], row['n'])})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    md_path = out.with_suffix(".md")
    csv_path = out.with_suffix(".csv")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "reference", "n", "correct_avg", "correct_pct", "physical_electrical", "telemetry_and_sensing", "atmosphere", "other", "trials"])
        for label, _directory, rows in models:
            for row in rows:
                cells = row["cells"]
                writer.writerow(
                    [
                        label,
                        row["reference"],
                        f"{row['n']:.4f}",
                        f"{row['correct']:.4f}",
                        f"{(row['correct'] / row['n'] * 100.0) if row['n'] else 0.0:.4f}",
                        f"{cells['physical_electrical']:.4f}",
                        f"{cells['telemetry_and_sensing']:.4f}",
                        f"{cells['atmosphere']:.4f}",
                        f"{cells[OTHER]:.4f}",
                        row["trials"],
                    ]
                )

    print("\n".join(lines))
    print(f"Saved: {md_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
