# Sollar Panel

Sollar Panel is a documentation-first blueprint for an end-to-end IoT and machine learning pipeline that starts on a Raspberry Pi Zero 2 W and ends with a containerized backend, live web dashboard, and dataset export flow for model training.

The current repository state is an architecture scaffold. It captures the edge-device requirements, server-side design, deployment constraints, and contributor rules before implementation starts.

## Features

- Resilient Raspberry Pi edge acquisition design for ADS1115 over I2C.
- MQTT-based ingestion with reconnect-safe behavior for unstable Wi-Fi and flaky sensor wiring.
- Time-series storage plan with `InfluxDB` as the default recommendation for a low-memory server.
- Feature engineering plan for deltas, rolling variance, and a heuristic illumination index.
- Daily dataset export path for Jupyter-based model training.
- Lightweight API and live dashboard plan with strict Docker isolation.
- Cloudflare Tunnel routing guidance that keeps the existing public site intact.

## Documentation

- [Project overview](docs/project-overview.md)
- [Edge device pipeline](docs/edge-device-pipeline.md)
- [Server pipeline](docs/server-pipeline.md)
- [Cloudflare Tunnel integration](docs/cloudflare-tunnel.md)
- [Repository operating rules](AGENT.md)

## Quickstart

The server scaffold is now ready for a first review pass. Copy the environment template and start the containers:

```bash
cd /home/tim/projects/sollar_panel
cp server/.env.example server/.env
docker compose --env-file server/.env -f server/docker-compose.yml up --build -d
docker compose --env-file server/.env -f server/docker-compose.yml ps
```

Default published ports:

- MQTT broker on `1884`
- InfluxDB UI and API on `127.0.0.1:18086`
- FastAPI scaffold on `127.0.0.1:18000`
- Frontend scaffold on `127.0.0.1:13000`

The repository now also includes an edge node implementation:

```bash
cp edge/.env.example edge/.env
python3 -m pip install -r edge/requirements.txt
python3 edge/solar_node.py
```

## Configuration

The planned runtime stack is:

- `mosquitto` for MQTT
- `influxdb` for time-series storage
- Python workers for ingestion and daily export
- `FastAPI` for history and live data delivery
- A separate frontend container for charts and status UI

The server design assumes a strict isolation boundary from the already-running public website and a constrained host with about `3 GB` of free RAM.

Initial server layout:

- `server/docker-compose.yml`
- `server/.env.example`
- `server/mosquitto/config/mosquitto.conf`
- `server/backend/`
- `server/worker/`
- `server/frontend/`
- `server/data/exports/`

Edge layout:

- `edge/solar_node.py`
- `edge/.env.example`
- `edge/requirements.txt`
- `edge/systemd/sollar-panel-edge.service`

The MVP backend now exposes:

- `GET /api/history` for aggregated day data
- `GET /api/live` as an SSE stream for real-time telemetry
- `GET /api/status` for a compact runtime summary

The frontend is a single-file dashboard that proxies API traffic through Nginx and renders a real-time Chart.js line chart plus a heuristic status widget.

## Development

- Use English everywhere in the repository: code, comments, logs, API messages, UI text, and docs.
- Follow the workflow and operational rules in [AGENT.md](AGENT.md).
- When `README.md` needs to be revised, do it with the `github-readme` skill workflow.
- Keep documentation synchronized with behavior changes.

## Contributing

This project currently follows an internal, documentation-driven workflow. Before implementing or changing behavior:

1. Update the relevant spec in `docs/`.
2. Update `README.md` if onboarding or architecture expectations changed.
3. Update `AGENT.md` if the workflow or project rules changed.
4. Create a local commit for the change set and do not push unless the user explicitly asks for it.

## License

License terms have not been defined yet. Do not assume open-source usage rights until a license is added.
