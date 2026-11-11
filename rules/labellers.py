"""Ground-truth root-cause labellers and their required episode utilities."""


from __future__ import annotations


from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


FIELD_MIN_ABS = {
    "power": 12.0,
    "current": 0.9,
    "bus_voltage": 3.0,
}


@dataclass
class Episode:
    anomaly_type: str
    start_time: datetime
    start_payload: dict
    resolved_payload: Optional[dict]

    @property
    def key(self) -> Tuple[str, str]:
        return self.anomaly_type, _dt_key(self.start_time)

    @property
    def severity(self) -> Optional[str]:
        if self.resolved_payload and self.resolved_payload.get("severity") is not None:
            return str(self.resolved_payload.get("severity"))
        if self.start_payload.get("severity") is not None:
            return str(self.start_payload.get("severity"))
        return None

    @property
    def detection_context(self) -> dict:
        context = self.start_payload.get("detection_context")
        return context if isinstance(context, dict) else {}

    @property
    def end_time(self) -> Optional[datetime]:
        if self.resolved_payload:
            parsed = _parse_iso_datetime(self.resolved_payload.get("end_time") or self.resolved_payload.get("detected_at"))
            if parsed is not None:
                return parsed
        return _parse_iso_datetime(self.start_payload.get("end_time"))

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.resolved_payload is not None:
            value = _parse_float(self.resolved_payload.get("duration_seconds"), default=None)
            if value is not None:
                return value
        value = _parse_float(self.start_payload.get("duration_seconds"), default=None)
        if value is not None:
            return value
        end_time = self.end_time
        if end_time is None:
            return None
        return max(0.0, (end_time - self.start_time).total_seconds())

    @property
    def data_points(self) -> int:
        if self.resolved_payload is not None and self.resolved_payload.get("data_points") is not None:
            return _parse_int(self.resolved_payload.get("data_points"), default=0)
        return _parse_int(self.start_payload.get("data_points"), default=0)

    @property
    def observed_value(self) -> Optional[float]:
        return _parse_float(self.detection_context.get("observed_value"), default=None)

    @property
    def simulated_value(self) -> Optional[float]:
        return _parse_float(self.detection_context.get("simulated_value"), default=None)

    @property
    def absolute_diff(self) -> Optional[float]:
        return _parse_float(self.detection_context.get("absolute_diff"), default=None)

    @property
    def rmse(self) -> Optional[float]:
        return _parse_float(self.detection_context.get("rmse"), default=None)

    @property
    def field(self) -> str:
        context_field = self.detection_context.get("field")
        if isinstance(context_field, str) and context_field.strip():
            return _normalize_field_name(context_field)
        return _normalize_field_name(self.anomaly_type)


def _normalize_field_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "voltage":
        return "bus_voltage"
    return text


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _dt_key(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _relative_diff(observed: Optional[float], simulated: Optional[float]) -> Optional[float]:
    if observed is None or simulated is None:
        return None
    return abs(observed - simulated) / max(abs(simulated), 1.0)


def _field_min_abs(field: str) -> float:
    return FIELD_MIN_ABS.get(field, 3.0)


def classify_episode_ground_truth(episode: Episode, llm_context: Optional[dict] = None) -> Dict[str, Any]:
    anomaly_type = _normalize_field_name(episode.anomaly_type)
    field = episode.field
    data_points = episode.data_points
    duration_seconds = episode.duration_seconds
    abs_diff = episode.absolute_diff
    rmse = episode.rmse
    observed = episode.observed_value
    simulated = episode.simulated_value
    rel_diff = _relative_diff(observed, simulated)
    moderate_threshold = _parse_float(episode.detection_context.get("moderate_threshold"), default=None)
    severe_threshold = _parse_float(episode.detection_context.get("severe_threshold"), default=None)
    context = llm_context.get("context") if isinstance(llm_context, dict) else {}
    if not isinstance(context, dict):
        context = {}
    inference_hints = context.get("inference_hints") if isinstance(context.get("inference_hints"), list) else []
    weather_latest = context.get("weather_latest") if isinstance(context.get("weather_latest"), dict) else {}
    baseline_same_hour = context.get("baseline_same_hour") if isinstance(context.get("baseline_same_hour"), dict) else {}

    evidence: List[str] = []

    if anomaly_type.startswith("missing_"):
        if anomaly_type.endswith("current_sensors") or anomaly_type.endswith("power_sensors"):
            root = "communication"
            confidence = 0.93
            evidence.append("missing_current_or_power_sensors_match_telemetry_transport_issue")
        else:
            root = "sensor_fault"
            confidence = 0.9
            evidence.append("missing_voltage_sensor_matches_sensor_fault_pattern")
        evidence.append(f"anomaly_type={anomaly_type}")
        if duration_seconds is not None:
            evidence.append(f"duration_seconds={duration_seconds:.1f}")
        return {
            "root_cause_category": root,
            "confidence": confidence,
            "evidence": evidence,
            "recommended_action": "Verify the measurement chain, telemetry path, and sensor health for the missing channel.",
        }

    if data_points <= 4 or (duration_seconds is not None and duration_seconds <= 4.0):
        if "weather_more_likely_than_hardware_fault" in inference_hints:
            root = "weather"
            confidence = 0.68
            evidence.append("llm_context_hint_prefers_weather_for_short_transient")
            if weather_latest:
                evidence.append(f"weather_latest_ghi={weather_latest.get('ghi')}")
                evidence.append(f"weather_latest_dni={weather_latest.get('dni')}")
            return {
                "root_cause_category": root,
                "confidence": confidence,
                "evidence": evidence[:6],
                "recommended_action": "Check whether irradiance changes are driving the transient anomaly before treating it as a hardware fault.",
            }
        evidence.append(f"insufficient_samples:data_points={data_points}")
        if duration_seconds is not None:
            evidence.append(f"short_duration_seconds={duration_seconds:.1f}")
        return {
            "root_cause_category": "unknown",
            "confidence": 0.2,
            "evidence": evidence,
            "recommended_action": "Collect a longer anomaly window before assigning a root cause.",
        }

    if field not in FIELD_MIN_ABS:
        evidence.append(f"unsupported_field={field}")
        return {
            "root_cause_category": "unknown",
            "confidence": 0.3,
            "evidence": evidence,
            "recommended_action": "Inspect the anomaly context and extend the baseline rules for this field.",
        }

    min_abs = _field_min_abs(field)
    severe = False
    moderate = False
    if rel_diff is not None and rel_diff >= 0.40:
        severe = True
        evidence.append(f"relative_diff={rel_diff:.3f}>=0.40")
    if abs_diff is not None and abs_diff >= 2.5 * min_abs:
        severe = True
        evidence.append(f"absolute_diff={abs_diff:.3f}>=2.5*min_abs({min_abs:.3f})")
    if severe_threshold is not None and abs_diff is not None and abs_diff > severe_threshold:
        severe = True
        evidence.append(f"absolute_diff={abs_diff:.3f}>severe_threshold={severe_threshold:.3f}")

    if rel_diff is not None and rel_diff >= 0.15:
        moderate = True
        evidence.append(f"relative_diff={rel_diff:.3f}>=0.15")
    if abs_diff is not None and abs_diff >= min_abs:
        moderate = True
        evidence.append(f"absolute_diff={abs_diff:.3f}>=min_abs({min_abs:.3f})")
    if moderate_threshold is not None and abs_diff is not None and abs_diff > moderate_threshold:
        moderate = True
        evidence.append(f"absolute_diff={abs_diff:.3f}>moderate_threshold={moderate_threshold:.3f}")

    if severe:
        if duration_seconds is not None and duration_seconds >= 60.0:
            if "persistent_anomaly" in inference_hints and field == "bus_voltage" and abs_diff is not None and abs_diff >= 3.0:
                root = "model_mismatch"
                confidence = 0.72
                evidence.append("llm_context_hint_prefers_model_mismatch_for_persistent_voltage_gap")
            else:
                root = "electrical_fault"
                confidence = 0.92
                evidence.append("long_duration_and_severe_threshold_match_electrical_fault")
        elif data_points < 10 and duration_seconds is not None and duration_seconds < 60.0:
            root = "weather"
            confidence = 0.55
            evidence.append("brief_severe_episode_kept_as_weather_like_transient")
        else:
            root = "electrical_fault"
            confidence = 0.86
            evidence.append("threshold_based_severe_deviation_matches_electrical_fault")
    elif moderate:
        if 5 <= data_points < 10 and duration_seconds is not None and duration_seconds < 60.0:
            root = "weather"
            confidence = 0.58
            evidence.append("short_moderate_episode_treated_as_weather_like_transient")
        else:
            root = "electrical_fault"
            confidence = 0.74
            evidence.append("moderate_threshold_episode_matches_electrical_fault_baseline")
    else:
        root = "unknown"
        confidence = 0.25
        evidence.append("no_threshold_rule_triggered")

    if rel_diff is not None:
        evidence.append(f"relative_diff={rel_diff:.3f}")
    if abs_diff is not None:
        evidence.append(f"absolute_diff={abs_diff:.3f}")
    if rmse is not None:
        evidence.append(f"rmse={rmse:.3f}")
    if duration_seconds is not None:
        evidence.append(f"duration_seconds={duration_seconds:.1f}")
    evidence.append(f"field={field}")

    return {
        "root_cause_category": root,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "evidence": evidence[:6],
        "recommended_action": "Inspect the electrical generation chain, expected-vs-observed deviation, and whether the episode is transient or sustained.",
    }


def classify_episode_ground_truth_without_hints(
    episode: Episode, llm_context: Optional[dict] = None
) -> Dict[str, Any]:
    """Apply the authoritative rules as if inference hints were empty.

    Preserve all other context and leave the caller's context unchanged.
    """
    context_wrapper = dict(llm_context) if isinstance(llm_context, dict) else {}
    context = context_wrapper.get("context")
    if isinstance(context, dict):
        context_wrapper["context"] = {
            key: value for key, value in context.items() if key != "inference_hints"
        }
    return classify_episode_ground_truth(episode, llm_context=context_wrapper)
