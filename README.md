# Sollar Panel

Sollar Panel is an end-to-end IoT and machine learning pipeline for a Pringles-can solar node. The edge baseline is now an ESP32 that samples solar voltage, reads temperature and humidity, updates a small OLED display, and streams MQTT telemetry into a containerized backend with live analytics and online ML.

The current repository state is a migration-ready MVP: the server stack is already live, the new ESP32 firmware publishes the same solar payload shape the backend expects, and the backend can now persist optional `temperature_c`, `humidity_pct`, and `adc_raw` fields alongside the existing voltage telemetry.

## Features

- ESP32 edge firmware with `GPIO34` solar sampling, `DHT11` climate readings, and SSD1306 OLED output.
- MQTT-based ingestion with reconnect-safe behavior for unstable Wi-Fi and fragile hardware wiring.
- Edge-side batching that keeps local sampling at `5 Hz` while publishing 1-second aggregates.
- Time-series storage plan with `InfluxDB` as the default recommendation for a low-memory server.
- Phase-aware online training service with feature engineering and two-stage models every 15 minutes.
- AI Insights for a light-only solar clock, sunset ETA, sunrise ETA, bias tracking, and confidence scoring.
- Command-center dashboard with live volatility, delta, residuals, percentiles, SNR, and uptime.
- Lightweight FastAPI surface for history, live telemetry, AI insights, and analytics summaries.
- Cloudflare Tunnel routing guidance that keeps the existing public site intact.

## Documentation

- [Project overview](docs/project-overview.md)
- [Edge device pipeline](docs/edge-device-pipeline.md)
- [Server pipeline](docs/server-pipeline.md)
- [Cloudflare Tunnel integration](docs/cloudflare-tunnel.md)
- [Repository operating rules](AGENT.md)

## Quickstart

Copy the environment template and start the server stack:

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

The repository includes a new ESP32 edge firmware target:

```bash
cd /home/tim/projects/sollar_panel/edge/esp32
cp include/secrets.example.h include/secrets.h
# fill in Wi-Fi and MQTT values inside include/secrets.h
# then build and flash with PlatformIO from your development machine
```

The default ESP32 edge runtime behavior is:

- sample `GPIO34` five times per second
- read `DHT11` on `GPIO4`
- refresh the OLED over I2C on `GPIO18` and `GPIO19`
- publish one aggregate MQTT packet every second
- keep the MQTT payload compatible with the existing ingestion worker

## Configuration

The runtime stack is:

- `mosquitto` for MQTT
- `influxdb` for time-series storage
- Python workers for ingestion and daily export
- `ml_engine` for online model training and model registry refresh
- `FastAPI` for history, analytics, and live data delivery
- A separate frontend container for the command-center UI

The server design assumes a strict isolation boundary from the already-running public website and a constrained host with about `3 GB` of free RAM.

Initial server layout:

- `server/docker-compose.yml`
- `server/.env.example`
- `server/mosquitto/config/mosquitto.conf`
- `server/backend/`
- `server/worker/`
- `server/ml_engine/`
- `server/frontend/`
- `server/data/exports/`

Edge layout:

- `edge/esp32/platformio.ini`
- `edge/esp32/include/secrets.example.h`
- `edge/esp32/src/main.cpp`

The backend now exposes:

- `GET /api/history` for aggregated day data
- `GET /api/live` as an SSE stream for real-time telemetry
- `GET /api/insights` for online-training output and confidence scores
- `GET /api/analytics` for last-hour percentiles, SNR, uptime, live volatility, delta, residual series, latest engineered features, and phase prediction
- `GET /api/status` for a compact runtime summary

Telemetry storage now accepts:

- solar voltage fields
- `temperature_c`
- `humidity_pct`
- `adc_raw`

The frontend is a single-file dashboard that proxies API traffic through Nginx and renders:

- a daily voltage area chart
- a 60-second live volatility oscilloscope
- a delta-per-second chart
- an adaptive AI residuals chart
- percentile, SNR, uptime, and confidence cards
- a bias card for the last-hour solar-clock drift
- an AI Insights panel for a light-only solar-clock estimate, sunset ETA, sunrise ETA, and ML phase context

The current ML stack includes:

- a phase classifier for `Night`, `Sunrise`, `Day`, `Sunset`, and `Anomaly`
- a time-of-day regressor for the AI solar clock estimate
- a day/sunset regressor for sunset ETA
- a night/sunrise regressor for sunrise ETA
- engineered features such as raw voltage, smoothed voltage, rolling standard deviation, multi-window deltas, and `voltage_to_daily_max_ratio`
- no `hour` or `minute` inputs for any live model, so the solar-clock estimate is driven purely by light behavior
- a 5-second Influx downsampling step for training so the ML engine stays lightweight even when live ingestion runs at `1 Hz`

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

Live site: https://solar.onewordtext.tech
