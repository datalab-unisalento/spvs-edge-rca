"""Extracted verbatim from the classifier; no LLM invocation."""

def _parse_float(value, default=None):
	try:
		if value is None:
			return default
		return float(value)
	except (TypeError, ValueError):
		return default

def _parse_int(value, default=0):
	try:
		if value is None:
			return default
		return int(value)
	except (TypeError, ValueError):
		return default

def _derive_root_cause_hints(context: dict):
	hints = []
	detection_context = context.get("detection_context") or {}
	history = context.get("history") or {}
	simulation = history.get("simulation") or {}
	weather = history.get("weather") or {}
	event = context.get("event") or {}

	absolute_diff = _parse_float(detection_context.get("absolute_diff"), default=None)
	moderate_threshold = _parse_float(detection_context.get("moderate_threshold"), default=None)
	data_points = _parse_int(event.get("data_points"), default=0)
	duration_seconds = _parse_float(event.get("duration_seconds"), default=0.0)

	temp_summary = weather.get("temp_air") or simulation.get("temp_air")
	humidity_summary = None
	wind_summary = weather.get("wind_speed") or simulation.get("wind_speed")

	if temp_summary and abs(float(temp_summary.get("trend_per_step", 0.0))) > 0.2:
		hints.append("temperature_trend_change_detected")
	if wind_summary and abs(float(wind_summary.get("trend_per_step", 0.0))) > 0.2:
		hints.append("wind_trend_change_detected")

	if (
		absolute_diff is not None
		and moderate_threshold is not None
		and absolute_diff <= 1.5 * moderate_threshold
		and ("temperature_trend_change_detected" in hints or "wind_trend_change_detected" in hints)
	):
		hints.append("weather_more_likely_than_hardware_fault")

	if data_points >= 8 or duration_seconds >= 300:
		hints.append("persistent_anomaly")
	if "persistent_anomaly" in hints and not (
		"temperature_trend_change_detected" in hints
		or "wind_trend_change_detected" in hints
	):
		hints.append("soiling_or_model_bias_candidate")

	return hints
