# Server Pipeline

## Goal

The server side receives solar telemetry from MQTT, stores the raw signal safely, computes features for future machine learning work, and serves both live and historical views to a web frontend.

## Database Recommendation

### Recommended Default: InfluxDB

Use `InfluxDB` as the default time-series database for the first implementation.

Reasons:

- lower operational complexity than a PostgreSQL plus TimescaleDB stack
- good fit for append-heavy telemetry
- easier to keep lightweight on a server with about `3 GB` of free RAM
- fast enough for dashboard queries and daily exports at the current project scale

### When to Reconsider TimescaleDB

TimescaleDB becomes more attractive if the project later needs:

- heavier relational joins
- complex SQL analytics across multiple entity types
- broader PostgreSQL ecosystem integration

For the first production version, `InfluxDB` is the better fit.

## Docker Compose Topology

The planned stack should live in one `docker-compose.yml` and remain isolated from the existing public website.

Recommended services:

- `mosquitto`
- `influxdb`
- `ingestor`
- `api`
- `ml_engine`
- `frontend`
- optional `exporter` or scheduled job container

To reduce maintenance and image sprawl on a small server:

- prefer one shared Python codebase image with different commands for `ingestor`, `api`, and `exporter`

### Initial Repository Layout

The first committed scaffold uses:

- `server/docker-compose.yml`
- `server/.env.example`
- `server/mosquitto/config/mosquitto.conf`
- `server/backend/`
- `server/worker/`
- `server/ml_engine/`
- `server/frontend/`
- `server/data/exports/`

### Default Host Port Plan

The initial compose file publishes:

- MQTT on host port `1884`
- InfluxDB on `127.0.0.1:18086`
- FastAPI on `127.0.0.1:18000`
- Frontend on `127.0.0.1:13000`

This keeps the web-facing services local to the server for now, while still allowing the edge device to reach MQTT over the LAN.

## MQTT Ingestion

### Broker

Run `Mosquitto` in Docker.

### Ingestion Worker Responsibilities

The ingestion worker must:

- subscribe to `sensor/solar/voltage`
- validate incoming payloads
- write `raw_voltage` to the time-series database as the canonical stored signal
- optionally store `smoothed_voltage` as a secondary field for UI convenience
- handle reconnects to both MQTT and the database without process crashes

### Reliability Requirements

Production-ready behavior should include:

- MQTT reconnect handling
- database write retry handling
- bounded logging that surfaces failures without flooding the disk
- safe behavior during malformed payloads

### Initial InfluxDB Structure

The first storage contract is:

- organization: `sollar_panel`
- bucket: `solar_metrics`
- measurement: `solar_voltage`

Initial tags:

- `sensor_id`

Initial fields:

- `raw_voltage`
- `smoothed_voltage` when present

The timestamp comes from the payload when available and falls back to current UTC when missing or invalid.

## Feature Engineering

The first feature set should include:

- `delta_v_5s`
- `delta_v_1m`
- `rolling_variance`
- `illumination_index`

### Implemented Online-Training Features

The current `ml_engine` implementation computes:

- `hour`
- `minute`
- `volt_velocity`
- `rolling_mean_5min`
- `effective_voltage` as the base training signal

Training cadence:

- every 15 minutes

Current model choices:

- lightweight `LinearRegression` models to respect the small memory budget

Current targets:

- local time-of-day estimate
- `time_to_sunset`
- `time_to_sunrise`

Model artifacts and latest insight snapshots are written into a shared `/models` volume.

### Suggested Semantics

- `delta_v_5s`: current `raw_voltage` minus the value from 5 seconds earlier
- `delta_v_1m`: current `raw_voltage` minus the value from 60 seconds earlier
- `rolling_variance`: variance over a recent window used to capture instability, twilight conditions, or noise
- `illumination_index`: a heuristic score derived from voltage level and recent stability until an ML model replaces it

## Daily Export Pipeline

Add a daily job that aggregates or extracts data into files for notebook work.

Recommended outputs:

- `CSV` for universal compatibility
- optional `Parquet` when columnar storage becomes useful

Recommended behavior:

- run once per day
- export a bounded time range
- preserve timestamps and feature columns
- write deterministic filenames
- avoid large in-memory processing when a streamed export is possible

## API Layer

Use `FastAPI` for the backend.

The API should expose:

- historical query endpoints
- summary endpoints for the current day
- a live data stream using `WebSockets` or `SSE`

Recommended initial API surface:

- `GET /api/history`
- `GET /api/summary/today`
- `GET /api/status`
- live stream endpoint for new points

### Implemented MVP API Surface

The current scaffold now includes:

- `GET /api/history`
- `GET /api/live`
- `GET /api/insights`
- `GET /api/status`
- `GET /healthz`

`/api/live` uses Server-Sent Events so the frontend can stream updates without reloading the page.

`/api/history` returns aggregated day data from InfluxDB with a configurable interval in minutes.

`/api/insights` serves the latest ML snapshot written by `ml_engine`.

### Timezone Handling

The system now treats `Europe/Copenhagen` as the application timezone for:

- day-boundary queries
- frontend chart labels
- ML feature extraction
- AI insight presentation

Stored telemetry remains in UTC inside InfluxDB. This avoids rewriting historical points or creating artificial gaps when changing timezone behavior.

## Frontend

The frontend should be lightweight and visually polished.

Requirements:

- dark theme
- gradient-based charts
- current voltage card
- day trend chart
- live updates
- status badge driven by heuristics for now

### Implemented MVP Frontend

The current frontend is intentionally kept in a single `index.html` file for simple deployment.

It includes:

- same-origin API access through Nginx proxying to the backend container
- a large real-time Chart.js line chart
- a current voltage panel
- a live connection state indicator
- a heuristic condition badge based on voltage thresholds
- an AI Insights panel for predicted local time, sunset ETA, sunrise ETA, and confidence level

Suggested status labels:

- `Sunny`
- `Cloudy`
- `Shade`

These are temporary heuristics and should be designed so an ML model can replace the classifier later without redesigning the UI contract.

## Production Readiness Expectations

The server-side implementation should be ready for long-running operation:

- reconnect safely after MQTT interruptions
- reconnect safely after DB interruptions
- isolate the stack from the existing site
- remain small enough for the host memory budget
- expose enough observability to debug failures without attaching a debugger
