# questa versione genera dati basati sulla simulazione e non sui dati vecchi
import pandas as pd
import time
import datetime
import os
import http.server
import threading
from pathlib import Path
from paho.mqtt import client as mqttc
from pvsimtools.PVSim import PVSim
import json
import numpy as np
import yaml

np.seterr(divide='ignore', invalid='ignore')
def _load_common_config():
    root_dir = Path(__file__).resolve().parents[1]
    config_path = Path(os.environ.get("SPVS_COMMON_CONFIG", root_dir / "config" / "systems.yaml"))
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


COMMON_CONFIG = _load_common_config()
MQTT_CLIENT_ID = COMMON_CONFIG["clients"]["spvs_emulator_rt"]
MQTT_BROKER_IP = COMMON_CONFIG["mqtt"]["broker_ip"]
MQTT_BROKER_PORT = COMMON_CONFIG["mqtt"]["broker_port"]
MQTT_TOPIC_SENSORS = COMMON_CONFIG["topics"]["smartpv"]["sensors"]
MQTT_TOPIC_POSITION = COMMON_CONFIG["topics"]["smartpv"]["position"]
MQTT_TOPIC_COMMANDS = COMMON_CONFIG["topics"]["smartpv"]["commands"]
MQTT_TOPIC_POSE = "smartpv/pose"
MQTT_PUBLISH_DELAY = 1
HEALTH_PORT = int(os.environ.get("SPVS_EMULATOR_RT_HEALTH_PORT", "18085"))

ANOMALY_ENABLED = True
# Random anomaly duration range (inclusive): number of data points in an anomaly
ANOMALY_DURATION_MIN_DATA_POINTS = 5
ANOMALY_DURATION_MAX_DATA_POINTS = 200

# Probability to start a new anomaly at each data point when no anomaly is active
ANOMALY_START_PROBABILITY = 0.005

# Minimum number of anomaly-free data points required before a new anomaly can start
ANOMALY_MIN_DATA_POINTS_WITHOUT_ANOMALY = 50

# Type probabilities used when an anomaly starts
ANOMALY_THRESHOLD_PROBABILITY = 0.5
ANOMALY_NULL_PROBABILITY = 0.5

# Threshold anomaly intensity
ANOMALY_THRESHOLD_DELTA_PARAMS = {
    # mean_ratio/std_ratio are relative to the current field value.
    # min_abs guarantees a minimum perturbation that still triggers anomalies.
    "power": {"mean_ratio": 0.25, "std_ratio": 0.08, "min_abs": 12.0},
    "current": {"mean_ratio": 0.30, "std_ratio": 0.10, "min_abs": 0.9},
    "bus_voltage": {"mean_ratio": 0.22, "std_ratio": 0.07, "min_abs": 3.0},
}
ANOMALY_THRESHOLD_MAX_RATIO = 0.60

# Threshold anomaly field candidates (payload keys in panel object)
ANOMALY_THRESHOLD_FIELDS = ["bus_voltage"]

# Null anomaly field candidates (payload keys in panel object)
ANOMALY_NULL_FIELDS = ["power", "current", "bus_voltage"]

SIMULATION_TIME_DELTA = 0

SENT_DATA_POINTS = 0
ACTIVE_ANOMALY_TYPE = None
ACTIVE_ANOMALY_REMAINING_POINTS = 0
ACTIVE_NULL_FIELD = None
ACTIVE_THRESHOLD_FIELD = None
DATA_POINTS_WITHOUT_ANOMALY = 0
STATIC_POSE_VALUES = {}
CURRENT_POSE_VALUES = {}
DESIRED_POSE_VALUES = {}


def _normalize_tilt(pitch_deg: float) -> float:
    # pvlib expects degrees from horizontal, bounded in [0, 180].
    return float(np.clip(pitch_deg, 0.0, 180.0))


def _normalize_azimuth(yaw_deg: float) -> float:
    # pvlib expects compass azimuth: N=0, E=90, S=180, W=270.
    return float(yaw_deg % 360.0)


def _mqtt_health_check():
    client = globals().get("mqtt_client")
    connected = bool(client and client.is_connected())
    return connected, "mqtt connected" if connected else "mqtt disconnected"


def _health_payload():
    mqtt_ok, mqtt_message = _mqtt_health_check()
    pvsim_ok = globals().get("pvsim") is not None
    checks = {
        "mqtt": {"ok": mqtt_ok, "message": mqtt_message},
        "pvsim": {"ok": pvsim_ok, "message": "pvsim initialized" if pvsim_ok else "pvsim missing"},
    }
    return {
        "service": "spvs_emulator_rt",
        "status": "ok" if all(item["ok"] for item in checks.values()) else "degraded",
        "checks": checks,
    }


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return

        payload = _health_payload()
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200 if payload["status"] == "ok" else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_health_server(port: int):
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _log_latency_marker(stage: str, **fields):
    record = {
        "event": "latency_marker",
        "service": "spvs_emulator_rt",
        "stage": stage,
        "logged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    for key, value in fields.items():
        if value is not None:
            record[key] = value
    print("LATENCY_MARKER " + json.dumps(record, ensure_ascii=True, separators=(",", ":")))


def _extract_payload_timestamp(payload: str):
    try:
        parsed = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed.get("timestamp")

# Fixed startup defaults without loading robotarm JSON data.
STATIC_POSE_VALUES = {
    "axis_roll": 0.0,
    "axis_pitch": 0.0,
    "axis_yaw": 180.0,
    "accelerometer_x": 0.0,
    "accelerometer_y": 0.0,
    "accelerometer_z": 0.0,
    "gyroscope_x": 0.0,
    "gyroscope_y": 0.0,
    "gyroscope_z": 0.0,
    "magnetometer_x": 0.0,
    "magnetometer_y": 0.0,
    "magnetometer_z": 0.0,
}


def _sample_threshold_delta(field_name, current_value):
    """Sample a field-aware threshold delta from a Gaussian distribution."""
    params = ANOMALY_THRESHOLD_DELTA_PARAMS[field_name]
    base_value = abs(float(current_value))

    sampled_ratio = np.random.normal(params["mean_ratio"], params["std_ratio"])
    sampled_ratio = float(np.clip(sampled_ratio, 0.05, ANOMALY_THRESHOLD_MAX_RATIO))

    relative_delta = base_value * sampled_ratio
    return max(params["min_abs"], relative_delta)


def _step_towards(current_value: float, desired_value: float) -> float:
    if current_value == desired_value:
        return current_value

    step = float(np.random.randint(1, 4))
    if current_value < desired_value:
        return min(current_value + step, desired_value)
    return max(current_value - step, desired_value)


def _update_pose_towards_desired():
    CURRENT_POSE_VALUES["axis_pitch"] = _step_towards(
        CURRENT_POSE_VALUES["axis_pitch"], DESIRED_POSE_VALUES["axis_pitch"]
    )
    CURRENT_POSE_VALUES["axis_yaw"] = _step_towards(
        CURRENT_POSE_VALUES["axis_yaw"], DESIRED_POSE_VALUES["axis_yaw"]
    )


def connect_callback(client: mqttc.Client, userdata, flags, reason_code, properties):
    client.subscribe(MQTT_TOPIC_COMMANDS)


def message_callback(client: mqttc.Client, userdata, message: mqttc.MQTTMessage):
    try:
        command = json.loads(message.payload.decode())
    except json.JSONDecodeError:
        print("RECEIVED COMMAND: invalid JSON")
        return

    if "y_deg" in command:
        DESIRED_POSE_VALUES["axis_yaw"] = float(command["y_deg"])
    if "z_deg" in command:
        DESIRED_POSE_VALUES["axis_pitch"] = float(command["z_deg"])

    print("RECEIVED COMMAND: " + json.dumps(command))


mqtt_client = mqttc.Client(
    client_id=MQTT_CLIENT_ID, callback_api_version=mqttc.CallbackAPIVersion.VERSION2
)
mqtt_client.on_connect = connect_callback
mqtt_client.on_message = message_callback
mqtt_client.connect(MQTT_BROKER_IP, port=MQTT_BROKER_PORT)

panel_data_dict = json.loads(
    '{"panel": {"power": 0.0, "current": 0.0, "bus_voltage":0.0}}'
)

mqtt_client.loop_start()

_start_health_server(HEALTH_PORT)

# Keep one PVSim instance and update orientation for each timestep.
pvsim = PVSim()


def mqtt_publish_with_delay(client: mqttc.Client, topic, payload, delay=0, stage="mqtt_publish"):
    _log_latency_marker(
        stage,
        topic=topic,
        payload_timestamp=_extract_payload_timestamp(payload),
    )
    client.publish(topic, payload)
    if delay > 0:
        time.sleep(delay)


def _build_pose_payload(timestamp: int) -> str:
    payload = {
        "orientation": {
            "roll": CURRENT_POSE_VALUES["axis_roll"],
            "pitch": CURRENT_POSE_VALUES["axis_pitch"],
            "yaw": CURRENT_POSE_VALUES["axis_yaw"],
        },
        "accel": {
            "ax": STATIC_POSE_VALUES["accelerometer_x"],
            "ay": STATIC_POSE_VALUES["accelerometer_y"],
            "az": STATIC_POSE_VALUES["accelerometer_z"],
        },
        "gyro": {
            "gx": STATIC_POSE_VALUES["gyroscope_x"],
            "gy": STATIC_POSE_VALUES["gyroscope_y"],
            "gz": STATIC_POSE_VALUES["gyroscope_z"],
        },
        "mag": {
            "mx": STATIC_POSE_VALUES["magnetometer_x"],
            "my": STATIC_POSE_VALUES["magnetometer_y"],
            "mz": STATIC_POSE_VALUES["magnetometer_z"],
        },
        "timestamp": int(timestamp),
    }
    return json.dumps(payload)


def _build_position_payload(pvsim: PVSim) -> str:
    payload = {
        "coordinates": {
            "lat": float(pvsim.latitude),
            "lon": float(pvsim.longitude),
        }
    }
    return json.dumps(payload)


CURRENT_POSE_VALUES.update(STATIC_POSE_VALUES)
DESIRED_POSE_VALUES.update(
    {
        "axis_pitch": STATIC_POSE_VALUES["axis_pitch"],
        "axis_yaw": STATIC_POSE_VALUES["axis_yaw"],
    }
)


while True:
    date_to_sim = datetime.datetime.now() + datetime.timedelta(hours=SIMULATION_TIME_DELTA)
    timestamp = int(date_to_sim.timestamp())
    panel_data_dict["timestamp"] = timestamp

    _update_pose_towards_desired()

    pvsim.set_orientation(
        pitch_deg=_normalize_tilt(CURRENT_POSE_VALUES["axis_pitch"]),
        yaw_deg=_normalize_azimuth(CURRENT_POSE_VALUES["axis_yaw"]),
    )
    
    dp = pvsim.run_simulation_single_timestamp(date_to_sim)


    p_mp = dp.iloc[0]["p_mp"]
    i_sc = dp.iloc[0]["i_sc"]
    v_oc = dp.iloc[0]["v_oc"]

    if (
        ANOMALY_ENABLED
        and ACTIVE_ANOMALY_TYPE is None
        and DATA_POINTS_WITHOUT_ANOMALY >= ANOMALY_MIN_DATA_POINTS_WITHOUT_ANOMALY
    ):
        if np.random.random() < ANOMALY_START_PROBABILITY:
            ACTIVE_ANOMALY_REMAINING_POINTS = np.random.randint(
                ANOMALY_DURATION_MIN_DATA_POINTS,
                ANOMALY_DURATION_MAX_DATA_POINTS + 1,
            )

            total_type_probability = ANOMALY_THRESHOLD_PROBABILITY + ANOMALY_NULL_PROBABILITY
            threshold_probability = 0.5
            if total_type_probability > 0:
                threshold_probability = ANOMALY_THRESHOLD_PROBABILITY / total_type_probability

            if np.random.random() < threshold_probability:
                ACTIVE_ANOMALY_TYPE = "threshold"
                ACTIVE_NULL_FIELD = None
                ACTIVE_THRESHOLD_FIELD = np.random.choice(ANOMALY_THRESHOLD_FIELDS)
                print(
                    "ANOMALY STARTED | type=threshold"
                    f" | field={ACTIVE_THRESHOLD_FIELD}"
                    f" | timestamp={timestamp}"
                    f" | duration_points={ACTIVE_ANOMALY_REMAINING_POINTS}"
                )
            else:
                ACTIVE_ANOMALY_TYPE = "null"
                ACTIVE_NULL_FIELD = np.random.choice(ANOMALY_NULL_FIELDS)
                ACTIVE_THRESHOLD_FIELD = None
                print(
                    "ANOMALY STARTED | type=null"
                    f" | field={ACTIVE_NULL_FIELD}"
                    f" | timestamp={timestamp}"
                    f" | duration_points={ACTIVE_ANOMALY_REMAINING_POINTS}"
                )

    if ACTIVE_ANOMALY_TYPE == "threshold":
        if ACTIVE_THRESHOLD_FIELD == "power":
            p_mp -= _sample_threshold_delta("power", p_mp)
        elif ACTIVE_THRESHOLD_FIELD == "current":
            i_sc -= _sample_threshold_delta("current", i_sc)
        elif ACTIVE_THRESHOLD_FIELD == "bus_voltage":
            v_oc -= _sample_threshold_delta("bus_voltage", v_oc)

    panel_data_dict["panel"]["power"] = p_mp + np.random.normal(0, 0.005 * abs(p_mp))
    panel_data_dict["panel"]["current"] = i_sc + np.random.normal(0, 0.01 * abs(i_sc))
    panel_data_dict["panel"]["bus_voltage"] = v_oc + np.random.normal(
        0, 0.005 * abs(v_oc)
    )

    if ACTIVE_ANOMALY_TYPE == "null" and ACTIVE_NULL_FIELD is not None:
        panel_data_dict["panel"][ACTIVE_NULL_FIELD] = None

    mqtt_publish_with_delay(
        mqtt_client,
        payload=json.dumps(panel_data_dict),
        delay=MQTT_PUBLISH_DELAY,
        topic=MQTT_TOPIC_SENSORS,
        stage="sensor_publish",
    )

    mqtt_publish_with_delay(
        mqtt_client,
        payload=_build_pose_payload(timestamp),
        topic=MQTT_TOPIC_POSE,
        stage="pose_publish",
    )
    mqtt_publish_with_delay(
        mqtt_client,
        payload=_build_position_payload(pvsim),
        topic=MQTT_TOPIC_POSITION,
        stage="position_publish",
    )

    if ACTIVE_ANOMALY_TYPE is not None:
        ACTIVE_ANOMALY_REMAINING_POINTS -= 1
        DATA_POINTS_WITHOUT_ANOMALY = 0
        if ACTIVE_ANOMALY_REMAINING_POINTS <= 0:
            ended_type = ACTIVE_ANOMALY_TYPE
            ended_field = ACTIVE_NULL_FIELD if ended_type == "null" else ACTIVE_THRESHOLD_FIELD
            print(
                f"ANOMALY ENDED | type={ended_type}"
                f" | field={ended_field}"
                f" | timestamp={timestamp}"
            )
            ACTIVE_ANOMALY_TYPE = None
            ACTIVE_NULL_FIELD = None
            ACTIVE_THRESHOLD_FIELD = None
            ACTIVE_ANOMALY_REMAINING_POINTS = 0
    else:
        DATA_POINTS_WITHOUT_ANOMALY += 1

    SENT_DATA_POINTS += 1


mqtt_client.loop_stop()
