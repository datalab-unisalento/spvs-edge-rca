#!/usr/bin/env python3
"""Ablation study for the SPVS LLM root-cause classifier.

The original LM Studio log stream (02_lms_log_stream.log) contains, for each of
the 64 anomaly events, the full prompt that was sent to the model (input) and the
model's JSON diagnosis (output). This script replays the same 64 events while
removing parts of the *context* given to the LLM, then validates each
configuration with the rule-based root cause validation framework
(validation_root_cause/rule_based_root_cause_validation.py) keeping the ground
truth fixed (computed once from the full context).

Configurations
--------------
full        : original context (no ablation; original log outputs are reused)
no_hints    : everything except inference_hints
no_weather  : everything except history.weather and history.weather_latest
no_baseline : everything except history.baseline_same_hour
minimal     : only event + detection_context (no history, weather, hints)

Output layout (under --out-dir, default ablation_study/)
    logs/02_lms_log_stream_<config>.log   replayed LM Studio log (UTF-16LE + BOM)
    contexts/context_<config>.json        ablated context per event (for inspection)
    validation/<config>/summary.{json,md} validation framework outputs
    comparison.md / comparison.csv        per-config metric comparison
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LLM_LOG = ROOT_DIR / "data" / "logs" / "02_lms_log_stream.log"
DEFAULT_DETECTOR_LOG = ROOT_DIR / "data" / "logs" / "04_identificazione_anomalie.log"
DEFAULT_CLASSIFIER_LOG = ROOT_DIR / "data" / "logs" / "03_classificazione_anomalie.log"
DEFAULT_MACRO_CONFIG = ROOT_DIR / "rules" / "semantic_groups.yaml"
VALIDATION_SCRIPT = ROOT_DIR / "rules" / "Labeller.py"

ALLOWED_ROOT_CAUSE_CATEGORIES = (
    "electrical_fault",
    "environmental_effect",
    "degradation",
    "sensor_fault",
    "communication",
    "weather",
    "soiling",
    "model_mismatch",
    "orientation_issue",
    "unknown",
)

CONFIGS = ("full", "no_hints", "no_weather", "no_baseline", "minimal")

LOG_ENCODING = "utf-16"  # matches original: UTF-16LE with BOM, \n newlines


# ---------------------------------------------------------------------------
# Context ablation transforms
# ---------------------------------------------------------------------------
def _transform_full(context: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(context)


def _transform_no_hints(context: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(context)
    out.pop("inference_hints", None)
    return out


def _transform_no_weather(context: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(context)
    history = out.get("history")
    if isinstance(history, dict):
        history.pop("weather", None)
        history.pop("weather_latest", None)
    return out


def _transform_no_baseline(context: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(context)
    history = out.get("history")
    if isinstance(history, dict):
        history.pop("baseline_same_hour", None)
    return out


def _transform_minimal(context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": copy.deepcopy(context.get("event") or {}),
        "detection_context": copy.deepcopy(context.get("detection_context") or {}),
    }


TRANSFORMS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "full": _transform_full,
    "no_hints": _transform_no_hints,
    "no_weather": _transform_no_weather,
    "no_baseline": _transform_no_baseline,
    "minimal": _transform_minimal,
}

CONFIG_DESCRIPTIONS = {
    "full": "full context (original system)",
    "no_hints": "all but inference_hints",
    "no_weather": "all but history.weather / history.weather_latest",
    "no_baseline": "all but history.baseline_same_hour",
    "minimal": "only event + detection_context",
}


# ---------------------------------------------------------------------------
# Log parsing (mirrors validation framework behaviour)
# ---------------------------------------------------------------------------
def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-16")
    return text.lstrip("\ufeff")


def load_log_records(path: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    """Return (all_records, input_records, output_records) in file order."""
    records: List[dict] = []
    for line in read_text(path).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    inputs = [r for r in records if (r.get("data") or {}).get("type") == "llm.prediction.input"]
    outputs = [r for r in records if (r.get("data") or {}).get("type") == "llm.prediction.output"]
    if len(inputs) != len(outputs):
        raise ValueError(
            f"Log {path} has {len(inputs)} input records but {len(outputs)} output records."
        )
    return records, inputs, outputs


def extract_context(input_text: str) -> Optional[Tuple[Dict[str, Any], int, int]]:
    """Return (context_dict, json_start, json_end_absolute) from the input prompt."""
    marker = "Context JSON:"
    idx = input_text.find(marker)
    if idx < 0:
        return None
    json_start = input_text.find("{", idx)
    if json_start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        context, rel_end = decoder.raw_decode(input_text[json_start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(context, dict):
        return None
    return context, json_start, json_start + rel_end


def extract_messages(input_text: str) -> Tuple[str, str]:
    """Extract (system_content, user_content) from the templated input prompt."""
    think_marker = "<|think|>\n"
    turn_end = "<turn|>"
    user_marker = "<|turn>user\n"

    sys_think = input_text.find(think_marker)
    if sys_think < 0:
        sys_start = input_text.find("<|turn>system\n") + len("<|turn>system\n")
    else:
        sys_start = sys_think + len(think_marker)
    first_turn_end = input_text.find(turn_end, sys_start)
    if first_turn_end < 0:
        raise ValueError("Could not locate end of system message in input prompt.")
    system_content = input_text[sys_start:first_turn_end].strip()

    user_start = input_text.find(user_marker, first_turn_end)
    if user_start < 0:
        raise ValueError("Could not locate user message in input prompt.")
    user_content_start = user_start + len(user_marker)
    user_end = input_text.find(turn_end, user_content_start)
    if user_end < 0:
        raise ValueError("Could not locate end of user message in input prompt.")
    user_content = input_text[user_content_start:user_end].rstrip()
    return system_content, user_content


def replace_context(user_content: str, new_context: Dict[str, Any]) -> str:
    """Replace the 'Context JSON: {...}' block inside the user prompt."""
    marker = "Context JSON:"
    idx = user_content.find(marker)
    if idx < 0:
        raise ValueError("'Context JSON:' marker not found in user prompt.")
    json_start = user_content.find("{", idx)
    decoder = json.JSONDecoder()
    _, rel_end = decoder.raw_decode(user_content[json_start:])
    json_end = json_start + rel_end
    return (
        user_content[:json_start]
        + json.dumps(new_context, ensure_ascii=True)
        + user_content[json_end:]
    )


def rebuild_input_text(system_content: str, new_user_content: str) -> str:
    return (
        "<|turn>system\n<|think|>\n"
        + system_content
        + "<turn|>\n<|turn>user\n"
        + new_user_content
        + "<turn|>\n<|turn>model\n"
    )


# ---------------------------------------------------------------------------
# LM Studio re-inference (mirrors sistema_classificazione_anomalie)
# ---------------------------------------------------------------------------
def _json_schema_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "anomaly_root_cause",
            "schema": {
                "type": "object",
                "properties": {
                    "root_cause": {"type": "string"},
                    "root_cause_category": {"type": "string", "enum": list(ALLOWED_ROOT_CAUSE_CATEGORIES)},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "recommended_action": {"type": "string"},
                },
                "required": [
                    "root_cause",
                    "root_cause_category",
                    "confidence",
                    "evidence",
                    "recommended_action",
                ],
                "additionalProperties": True,
            },
        },
    }


def call_llm(
    system_content: str,
    user_content: str,
    *,
    api_base: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
    """Send one chat-completions request. Returns (content, stats, error)."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    base_body = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    }
    if reasoning_effort:
        base_body["reasoning_effort"] = reasoning_effort
    format_attempts = [_json_schema_response_format(), None]
    last_error: Optional[str] = None
    for fmt in format_attempts:
        body = dict(base_body)
        body["model"] = model
        if fmt is not None:
            body["response_format"] = fmt
        start = time.time()
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            elapsed = time.time() - start
            choice = (result.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            if not content:
                last_error = "empty content returned"
                continue
            # The validation framework expects strict JSON with root_cause_category.
            try:
                parsed_content = json.loads(content)
                if not isinstance(parsed_content, dict) or "root_cause_category" not in parsed_content:
                    raise ValueError("output is not a JSON object with root_cause_category")
            except (json.JSONDecodeError, ValueError):
                last_error = "non-JSON output returned; retrying without response_format"
                continue
            usage = result.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            finish = choice.get("finish_reason")
            stop = {"stop": "eosFound", "length": "maxTokens", "tool_calls": "toolCalls"}.get(finish)
            stats = {
                "stopReason": stop or "eosFound",
                "tokensPerSecond": (completion_tokens / elapsed)
                if completion_tokens and elapsed
                else None,
                "numGpuLayers": -1,
                "timeToFirstTokenSec": None,
                "totalTimeSec": round(elapsed, 3),
                "promptTokensCount": prompt_tokens,
                "predictedTokensCount": completion_tokens,
                "totalTokensCount": total_tokens,
            }
            return content, stats, None
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {error_body}"
            if exc.code == 400 and fmt is not None:
                # Retry without response_format (LM Studio compatibility).
                continue
        except urllib.error.URLError as exc:
            last_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        # Fall through to next format attempt unless a connection-type error occurred.
        if last_error and not last_error.startswith("HTTP 400"):
            break
    return None, None, last_error


def build_fallback_output(error: str) -> str:
    return json.dumps(
        {
            "root_cause": "Root cause unavailable (LLM request failed)",
            "root_cause_category": "unknown",
            "confidence": 0.0,
            "evidence": [f"llm_error={error}"],
            "recommended_action": "Check LM Studio server availability, loaded model, and API compatibility.",
        },
        ensure_ascii=True,
    )


def build_output_record(
    timestamp: int,
    content: str,
    stats: Optional[dict],
    model_identifier: str,
) -> dict:
    return {
        "timestamp": timestamp,
        "data": {
            "type": "llm.prediction.output",
            "output": content,
            "stats": stats or {},
            "modelIdentifier": model_identifier,
        },
    }


def build_input_record(timestamp: int, input_text: str, model_path: str, model_identifier: str) -> dict:
    return {
        "timestamp": timestamp,
        "data": {
            "type": "llm.prediction.input",
            "input": input_text,
            "modelPath": model_path,
            "modelIdentifier": model_identifier,
        },
    }


# ---------------------------------------------------------------------------
# Log writing
# ---------------------------------------------------------------------------
def write_log(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=LOG_ENCODING, newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_contexts(path: Path, contexts: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contexts, indent=2, ensure_ascii=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def run_validation(
    config_log: Path,
    ground_truth_log: Path,
    out_dir: Path,
    detector_log: Path,
    classifier_log: Path,
    macro_config: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(VALIDATION_SCRIPT),
        "--detector-log", str(detector_log),
        "--classifier-log", str(classifier_log),
        "--llm-log", str(config_log),
        "--ground-truth-llm-log", str(ground_truth_log),
        "--macro-config", str(macro_config),
        "--output-dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"Validation did not produce summary.json for {config_log}.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------
def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.2f}%"


def _num(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}"


def compute_prompt_length_stats(config_log: Path) -> dict:
    """Mean/median input-prompt length for a configuration log.

    The prompt length in tokens is read from the per-output API usage
    (promptTokensCount); the length in characters is the mean length of the
    templated input prompt text stored in the log.
    """
    if not config_log.exists():
        return {
            "avg_prompt_tokens": None,
            "median_prompt_tokens": None,
            "avg_prompt_chars": None,
        }
    prompt_tokens: List[float] = []
    prompt_chars: List[float] = []
    for line in read_text(config_log).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = record.get("data") or {}
        record_type = data.get("type")
        if record_type == "llm.prediction.output":
            stats = data.get("stats") or {}
            if stats.get("promptTokensCount") is not None:
                prompt_tokens.append(float(stats["promptTokensCount"]))
        elif record_type == "llm.prediction.input":
            input_text = data.get("input")
            if isinstance(input_text, str):
                prompt_chars.append(float(len(input_text)))
    return {
        "avg_prompt_tokens": mean(prompt_tokens) if prompt_tokens else None,
        "median_prompt_tokens": median(prompt_tokens) if prompt_tokens else None,
        "avg_prompt_chars": mean(prompt_chars) if prompt_chars else None,
    }


def count_failed_calls(config_log: Path) -> int:
    """Count LLM predictions that fell back to the 'unavailable' placeholder.

    A failed call is an output record whose content is the fallback produced by
    build_fallback_output() (root cause unavailable / llm_error evidence).
    """
    if not config_log.exists():
        return 0
    failed = 0
    for line in read_text(config_log).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        data = record.get("data") or {}
        if data.get("type") != "llm.prediction.output":
            continue
        output = data.get("output") or ""
        if "Root cause unavailable" in output or "llm_error" in output:
            failed += 1
    return failed


def _summary_rows(summaries: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for config in CONFIGS:
        summary = summaries.get(config)
        if summary is None:
            continue
        llm_fine = summary.get("ground_truth_vs_llm_metrics") or {}
        llm_macro = summary.get("ground_truth_vs_llm_macro") or {}
        baseline_fine = summary.get("ground_truth_vs_baseline_metrics") or {}
        episodes = summary.get("episodes") or {}
        prompt_len = summary.get("prompt_length") or {}
        rows.append(
            {
                "config": config,
                "description": CONFIG_DESCRIPTIONS[config],
                "episodes": episodes.get("evaluated", ""),
                "coverage_pct": _pct(episodes.get("coverage_pct")),
                "llm_fine_accuracy": _pct(llm_fine.get("accuracy")),
                "llm_fine_macro_f1": _pct(llm_fine.get("macro_f1")),
                "llm_fine_balanced_accuracy": _pct(llm_fine.get("balanced_accuracy")),
                "llm_fine_kappa": f"{llm_fine.get('kappa', float('nan')):.3f}"
                if llm_fine.get("kappa") is not None
                else "n/a",
                "llm_macro_accuracy": _pct(llm_macro.get("accuracy")),
                "llm_macro_macro_f1": _pct(llm_macro.get("macro_f1")),
                "baseline_fine_accuracy": _pct(baseline_fine.get("accuracy")),
                "avg_prompt_tokens": _num(prompt_len.get("avg_prompt_tokens")),
                "median_prompt_tokens": _num(prompt_len.get("median_prompt_tokens")),
                "avg_prompt_chars": _num(prompt_len.get("avg_prompt_chars")),
            }
        )
    return rows


def write_comparison(out_dir: Path, summaries: Dict[str, dict]) -> None:
    rows = _summary_rows(summaries)
    csv_path = out_dir / "comparison.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        import csv as _csv

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = _csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    lines = ["# Root-Cause LLM Ablation Study", ""]
    lines.append("Ground truth and baseline are computed from the **full** context (fixed reference);")
    lines.append("only the LLM input context is ablated per configuration.")
    lines.append("")
    lines.append("| Config | Description | Fine acc. | Fine macro F1 | Fine bal. acc. | Fine kappa | Macro acc. | Macro F1 | Baseline fine acc. | Avg prompt tok | Med prompt tok | Avg prompt chars |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['description']} | {row['llm_fine_accuracy']} "
            f"| {row['llm_fine_macro_f1']} | {row['llm_fine_balanced_accuracy']} | {row['llm_fine_kappa']} "
            f"| {row['llm_macro_accuracy']} | {row['llm_macro_macro_f1']} | {row['baseline_fine_accuracy']} "
            f"| {row['avg_prompt_tokens']} | {row['median_prompt_tokens']} | {row['avg_prompt_chars']} |"
        )
    lines.append("")
    lines.append("## Details per configuration")
    lines.append("")
    for config in CONFIGS:
        summary = summaries.get(config)
        if summary is None:
            continue
        llm_fine = summary.get("ground_truth_vs_llm_metrics") or {}
        prompt_len = summary.get("prompt_length") or {}
        lines.append(f"### {config} — {CONFIG_DESCRIPTIONS[config]}")
        lines.append("")
        lines.append(f"- LLM fine accuracy: {_pct(llm_fine.get('accuracy'))}")
        lines.append(f"- LLM fine macro F1: {_pct(llm_fine.get('macro_f1'))}")
        lines.append(
            f"- Input prompt length (tokens): mean={_num(prompt_len.get('avg_prompt_tokens'))}, "
            f"median={_num(prompt_len.get('median_prompt_tokens'))}"
        )
        lines.append(f"- Input prompt length (chars): mean={_num(prompt_len.get('avg_prompt_chars'))}")
        per_class = llm_fine.get("per_class") or {}
        for label in llm_fine.get("labels", []):
            stats = per_class.get(label) or {}
            lines.append(
                f"  - {label}: precision={_pct(stats.get('precision'))}, recall={_pct(stats.get('recall'))}, "
                f"f1={_pct(stats.get('f1'))}, support={int(stats.get('support', 0))}"
            )
        mismatches = summary.get("ground_truth_vs_llm_mismatches") or []
        lines.append(f"- Mismatches (GT vs LLM): {len(mismatches)}")
        for row in mismatches[:10]:
            lines.append(
                f"  - {row.get('episode_key')}: ground_truth={row.get('ground_truth_category')} "
                f"llm={row.get('llm_category')}"
            )
        lines.append("")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison report written to {out_dir / 'comparison.md'}")
    if rows:
        print(f"Comparison CSV written to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SPVS root-cause LLM ablation study.")
    parser.add_argument("--llm-log", type=Path, default=DEFAULT_LLM_LOG,
                        help="Original LM Studio log stream (default: %(default)s).")
    parser.add_argument("--detector-log", type=Path, default=DEFAULT_DETECTOR_LOG)
    parser.add_argument("--classifier-log", type=Path, default=DEFAULT_CLASSIFIER_LOG)
    parser.add_argument("--macro-config", type=Path, default=DEFAULT_MACRO_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "ablation_study",
                        help="Output directory (default: %(default)s).")
    parser.add_argument("--configs", default=",".join(CONFIGS),
                        help="Comma-separated configs to run (default: all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Restrict to the first N events (smoke test).")
    parser.add_argument("--api-base", default="http://127.0.0.1:1234/v1",
                        help="LM Studio OpenAI-compatible API base URL.")
    parser.add_argument("--model", default=None,
                        help="Model identifier (default: taken from the log).")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--skip-inference", action="store_true",
                        help="Do not call the LLM; reuse existing config logs if present (full always reuses original outputs).")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip inference entirely and only run validation on existing config logs.")
    parser.add_argument("--retries", type=int, default=2,
                        help="Number of extra attempts per failed LLM call (default: %(default)s).")
    return parser.parse_args()


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

    # ---- Load original log and derive per-event data ---------------------
    print(f"Loading original log: {original_log}")
    _, input_records, output_records = load_log_records(original_log)
    if args.limit:
        input_records = input_records[: args.limit]
        output_records = output_records[: args.limit]

    model_identifier = str(
        (input_records[0].get("data") or {}).get("modelIdentifier") or args.model or "google/gemma-4-e4b"
    )
    model_path = str((input_records[0].get("data") or {}).get("modelPath") or model_identifier)
    if args.model:
        model_identifier = args.model
        model_path = args.model

    system_content: Optional[str] = None
    events: List[dict] = []
    for input_record, output_record in zip(input_records, output_records):
        input_text: str = (input_record.get("data") or {}).get("input") or ""
        parsed = extract_context(input_text)
        if parsed is None:
            raise SystemExit("Could not parse 'Context JSON:' from an input record.")
        context, json_start, json_end = parsed
        sys_content, user_content = extract_messages(input_text)
        if system_content is None:
            system_content = sys_content
        elif sys_content != system_content:
            raise SystemExit("System prompt differs across input records; aborting.")
        events.append(
            {
                "input_timestamp": input_record.get("timestamp"),
                "output_timestamp": output_record.get("timestamp"),
                "context": context,
                "user_content": user_content,
                "original_output": (output_record.get("data") or {}).get("output"),
                "original_stats": (output_record.get("data") or {}).get("stats"),
            }
        )
    n_events = len(events)
    print(f"Parsed {n_events} events (context + output). Model: {model_identifier}")

    summaries: Dict[str, dict] = {}

    for config in configs:
        print(f"\n=== Config: {config} ({CONFIG_DESCRIPTIONS[config]}) ===")
        config_log = logs_dir / f"02_lms_log_stream_{config}.log"
        context_file = contexts_dir / f"context_{config}.json"
        validation_out = validation_dir / config

        ablated_contexts = [TRANSFORMS[config](ev["context"]) for ev in events]
        write_contexts(context_file, ablated_contexts)

        new_records: List[dict] = []
        skip_write = False
        if config == "full":
            # Reuse the original log verbatim (no re-inference).
            if args.limit is None:
                import shutil

                shutil.copyfile(original_log, config_log)
                skip_write = True
                print(f"  full: copied original log verbatim ({n_events} events).")
            else:
                for inp, out in zip(input_records, output_records):
                    new_records.append(inp)
                    new_records.append(out)
                print(f"  full: reused {len(output_records)} original output records (limit).")
        elif args.validate_only:
            # Reuse an existing config log if present (it already contains the
            # input/output records); only rebuild input records for traceability
            # when no log exists yet.
            if config_log.exists():
                skip_write = True
                new_records = []
                print(f"  validate-only: reusing existing log {config_log}")
            else:
                new_records = []
                for ev in events:
                    new_records.append(build_input_record(
                        ev["input_timestamp"],
                        rebuild_input_text(system_content, replace_context(ev["user_content"], ev["context"])),
                        model_path,
                        model_identifier,
                    ))
                print(f"  validate-only: no existing log for {config}; validation will have 0% coverage.")
        elif args.skip_inference:
            # Reuse an existing config log if present; otherwise write ablated
            # input records only (no outputs -> 0% coverage).
            if config_log.exists():
                skip_write = True
                new_records = []
                print(f"  skip-inference: reusing existing log {config_log}")
            else:
                new_records = []
                for ev in events:
                    new_records.append(build_input_record(
                        ev["input_timestamp"],
                        rebuild_input_text(system_content, replace_context(ev["user_content"], TRANSFORMS[config](ev["context"]))),
                        model_path,
                        model_identifier,
                    ))
                print(f"  skip-inference: NO existing log for {config}; validation will have 0% coverage.")
        else:
            # Re-inference: send ablated user prompt to the model for every event.
            failures = 0
            for i, ev in enumerate(events, start=1):
                new_user = replace_context(ev["user_content"], ablated_contexts[i - 1])
                new_input_text = rebuild_input_text(system_content, new_user)
                new_records.append(build_input_record(
                    ev["input_timestamp"], new_input_text, model_path, model_identifier
                ))
                content: Optional[str] = None
                stats: Optional[dict] = None
                error: Optional[str] = None
                for attempt in range(args.retries + 1):
                    content, stats, error = call_llm(
                        system_content,
                        new_user,
                        api_base=args.api_base,
                        model=model_identifier,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                    if content is not None:
                        break
                    if error and error.startswith("HTTP 4") and attempt < args.retries:
                        continue
                    if error and not error.startswith("HTTP 4"):
                        break  # connection-level failure; retrying won't help immediately
                if content is None:
                    failures += 1
                    content = build_fallback_output(error or "unknown error")
                    stats = {"totalTimeSec": 0.0, "promptTokensCount": None,
                             "predictedTokensCount": None, "totalTokensCount": None}
                new_records.append(build_output_record(
                    int(time.time() * 1000) + i,
                    content,
                    stats,
                    model_identifier,
                ))
                if i % 10 == 0 or i == n_events:
                    print(f"    event {i}/{n_events} done ({config})")
            if failures:
                print(f"  WARNING: {failures}/{n_events} LLM calls failed for config '{config}'.")

        if not skip_write:
            write_log(config_log, new_records)
            print(f"  wrote {config_log} ({len(new_records)} records)")
        else:
            print(f"  log ready: {config_log}")

        # ---- Validate ------------------------------------------------------
        summary = run_validation(
            config_log=config_log,
            ground_truth_log=original_log,
            out_dir=validation_out,
            detector_log=detector_log,
            classifier_log=classifier_log,
            macro_config=macro_config,
        )
        summary["prompt_length"] = compute_prompt_length_stats(config_log)
        summaries[config] = summary
        llm = summary.get("ground_truth_vs_llm_metrics") or {}
        print(f"  validation -> fine acc {_pct(llm.get('accuracy'))}, "
              f"macro F1 {_pct(llm.get('macro_f1'))}")

    write_comparison(out_dir, summaries)
    print("\nAblation study complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
