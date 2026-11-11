## Preliminary flags

Severe = rel_diff >= 0.40 OR abs_diff >= 2.5 × min_abs(field) OR abs_diff > severe_threshold.

Moderate = rel_diff >= 0.15 OR abs_diff >= min_abs(field) OR abs_diff > moderate_threshold.

Minimum absolute values: power 12.0 W; current 0.9 A; bus_voltage 3.0 V. The historical rationale uses these as minimum meaningful deviations above normal noise and fluctuations; it provides no independent threshold calibration evidence.

## R1: missing data — highest priority

Condition: anomaly_type starts with missing_.

| Duration | Category | Confidence |
|---|---|---|
| <= 30 s | communication | 0.78 |
| 30–300 s | sensor_fault | 0.84 |
| > 300 s | sensor_fault | 0.92 |

The source describes the middle interval as 30–300 s while the first includes 30 s; the intended boundary requires checking against the historical code. Evidence: anomaly_type, duration_seconds and data_points. Inspect the measurement chain, telemetry path and missing channel's sensor. The stated rationale associates short gaps with MQTT packet loss, jitter or temporary transport interruption, and persistent gaps with sensor stalls or disconnection. Its assumption that MQTT interruptions rarely exceed 30 seconds is not independently substantiated here.

## R2: insufficient samples

Condition: data_points <= 4 OR duration_seconds <= 4.0. With weather_more_likely_than_hardware_fault in inference_hints: weather, confidence 0.65. Without that hint: unknown, confidence 0.20. Evidence: point count, duration, weather_latest_ghi and weather_latest_dni. Check irradiance changes before diagnosing hardware, or gather a longer window. The rationale is limited evidence for attribution in such short windows.

## R3: unsupported field

Condition: field outside power/current/bus_voltage in FIELD_MIN_ABS. Result: unknown, confidence 0.30. Evidence: unsupported_field=<field>. Inspect context and extend the baseline rules with appropriate thresholds. Other physical fields require separate thresholds.

## R4: tracker orientation issue

Pose data must be available, with pitch_delta > 10.0 degrees OR yaw_delta > 15.0 degrees. Deltas are abs(pitch_mean - pitch_expected) and abs(yaw_mean - yaw_expected). Result: orientation_issue, confidence 0.90. Evidence: pose_misalignment_detected_from_tracker_data and angular deltas. Inspect motorized tracker alignment and expected pose tables.

The stated rationale prioritizes orientation before electrical severity because misalignment can affect electrical measurements together. The source attributes thresholds to mechanical tolerances and claims smaller errors cause less than 2% power variation; these claims are not independently verified in this artifact.

## R5: transient weather

Evaluated only when moderate or severe is true.

| Branch | Condition | Confidence |
|---|---|---|
| R5a | weather_more_likely_than_hardware_fault hint and duration < 120 s | 0.72 |
| R5b | severe and data_points < 10 and duration < 60 s | 0.55 |
| R5c | moderate and not severe and 5 <= data_points < 10 and duration < 60 s | 0.58 |

All branches assign weather. Evidence respectively includes weather_hint_and_duration<120s_matches_transient_meteorological_effect with GHI/DNI; brief_severe_episode_kept_as_weather_like_transient; or short_moderate_episode_treated_as_weather_like_transient. Examine irradiance trends before diagnosing hardware. The source associates brief deviations with passing clouds, citing assumed fluctuation ranges of 30–180 s for cumulus clouds and 60–300 s for cloud layers, without independent verification here.

## R6: severe electrical fault

Condition: severe. Duration >= 60 s gives electrical_fault, confidence 0.92, evidence severe_and_duration>=60s_matches_electrical_fault. Remaining shorter severe cases not captured by R5b give electrical_fault, confidence 0.86, evidence threshold_based_severe_deviation_matches_electrical_fault. Inspect the generation chain, expected-versus-observed gap and persistence. The rationale prioritizes severe electrical faults over soiling and degradation because they may require faster intervention.

## R7: soiling

All conditions: moderate, soiling_or_model_bias_candidate hint, duration > 600 s, observed < simulated and no recent precipitation. Result: soiling, confidence 0.65. Evidence: negative_bias_with_dry_weather_and_duration>600s_matches_soiling and observed/simulated values. Inspect the panel surface and consider cleaning if confirmed. The rationale is persistent output reduction without rain-driven cleaning; visual inspection is needed to distinguish soiling from model mismatch.

## R8: environmental effect

Condition: moderate, duration > 120 s and at least one of air temperature > 40 degrees C, wind speed > 15 m/s or relative humidity > 95%. Result: environmental_effect, confidence 0.60. Evidence: extreme_site_conditions_with_moderate_persistent_deviation and weather values. Inspect site conditions and possible temperature/wind/humidity derating. The source associates these conditions with thermal loss, mechanical stress or condensation, while noting local weather data may not represent the panel precisely.

## R9: remaining moderate electrical fault

Condition: moderate and no preceding R4–R8 match. Result: electrical_fault, confidence 0.74. Evidence: moderate_threshold_episode_without_other_explanation_matches_electrical_fault. Inspect for emerging electrical problems. This is attribution by exclusion rather than direct fault evidence.

## R10: gradual degradation

All conditions: duration > 1800 s, baseline available, drift_ratio > 0.10 and not severe. drift_ratio = abs(observed_median - baseline_median)/abs(baseline_median). Result: degradation, confidence 0.55. Evidence: baseline_drift_detected_over_long_episode_matches_gradual_degradation and drift ratio. Monitor long-term trends; consider electroluminescence or I–V tests for PID/microcracks. The historical rationale compares a same-hour baseline to reduce daily and seasonal variation, but acknowledges ambiguity with model mismatch and changing operating conditions. Whether R9 masks some R10 cases depends on the historical implementation and input conditions.

## R11: fallback

No preceding rule matches: unknown, confidence 0.25. Evidence: no_rule_matched_in_ground_truth_classifier. Collect a longer window or additional context. Possible explanations include detector false positives, noise, unsupported fields or phenomena outside the rule set.

## R12: post-hoc cohort model mismatch

Apply _cohort_model_mismatch_overrides() after individual classification. At least three episodes must have initial ground_truth_category unknown, a supported field, the same field and episode_hour, median relative deviation >= 0.12 and spread (max - min) <= 0.20. Relabel unknown to model_mismatch with confidence 0.68, adding systematic_bias_across_episodes_detected.

The rationale is repeated same-hour bias indicating calibration, input parameter or model approximation error. Three episodes provide a minimum pattern; median and spread thresholds require substantial, consistent deviation. This historical document limits the override to unknown labels; the available source instead targets electrical_fault labels with additional exclusions.

## Stated design principles

Ordered first-match rules support reproducibility. Separate conditions cover missing telemetry, orientation, transient weather, severe electrical deviations, soiling, environmental effects and degradation. Insufficient evidence leads to low-confidence unknown. Severe electrical faults take priority over less urgent explanations. Cohort processing captures systematic patterns hidden at the individual-episode level. All confidence values above are heuristic assignments from this historical description.
