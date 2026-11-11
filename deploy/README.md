# Deployment

`compose.yaml` combines the OSS stack (Mosquitto, ClickHouse, Telegraf,
Alertmanager, Grafana and Node-RED) and the four Python real-time services.
Run from this directory within the full repository: image builds use the
repository root (`../..`), its Dockerfiles and shared library.

Copy `.env.example` to `.env`, supply the ClickHouse password and Telegram
credentials, then run:

```sh
python -B render_config.py
docker compose --env-file .env config --quiet
docker compose --env-file .env up -d --build
```


`systems.yaml` preserves the repository topics and client identifiers and
points Python services to the Compose MQTT broker. Internal connections use
service names: ClickHouse HTTP uses port 8123 (host port 8124).
ClickHouse data/logs, Grafana, Alertmanager and Node-RED use named volumes;
`docker compose down` preserves them.

Run LM Studio separately on the host, load the desired model and enable
access from containers. Set `LLM_MODEL` and `LLM_API_BASE_URL` in `.env`;
the default URL is `http://host.docker.internal:1234/v1`.

Published ports: MQTT 1883, ClickHouse HTTP 8124, Alertmanager 9093,
Grafana 3000, Node-RED 1880 and Python health endpoints 18080, 18081, 18083
and 18085.
Grafana installs plugins at startup; configure data sources in its UI using
internal service names. Mosquitto permits anonymous access for this local
experiment. Original image tags are retained, including floating tags.

Open the Node-RED editor at `http://localhost:1880`. The `nodered` service
starts after the MQTT broker is healthy and persists flows, settings and
installed nodes in the `nodered-data` volume mounted at `/data`.
To use the optional reference flow, import `nodered_flow.json` through the
editor.
