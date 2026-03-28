# AGENT.md

## Project Summary

This repository defines an end-to-end IoT and machine learning pipeline for a small solar-panel sensor system:

- Edge baseline: ESP32 running Arduino firmware.
- Sensor path: a small solar panel connected to `GPIO34` through the ESP32 ADC path.
- Additional edge hardware: `DHT11` on `GPIO4` and SSD1306 OLED on I2C `GPIO18`/`GPIO19`.
- Backend: Dockerized MQTT, time-series storage, ingestion workers, API, and frontend.
- Goal: Collect stable telemetry, preserve raw signal quality, and build a clean dataset for future ML models.

The ESP32 firmware path is the active edge baseline. New edge changes should target the ESP32 implementation first.

## Language Policy

Everything in this repository must be in English:

- source code
- comments
- logs
- commit messages
- API payload names
- UI labels and button text
- README and documentation

Do not add Russian text to tracked project files unless the user explicitly asks for a translated artifact.

## Change Workflow

After each completed change set:

1. Update code.
2. Update `README.md` if onboarding, structure, or behavior changed.
3. Update `docs/` if technical behavior, architecture, or deployment expectations changed.
4. Update `AGENT.md` if workflow rules or project constraints changed.
5. Create a local git commit.
6. Do not push unless the user explicitly requests a push.

If the workspace is not initialized as a git repository yet, initialize git before the first commit unless the user instructs otherwise.

## README Workflow

Whenever `README.md` is created or revised, use the `github-readme` skill workflow. Keep the README practical, value-first, and easy to onboard from.

## Infrastructure Notes

Two user-provided SSH targets exist for this project:

- one edge-device host
- one primary server host

Never commit or document their hostnames, IP addresses, usernames, passwords, private keys, or any other sensitive connection details in `README.md`, `docs/`, `AGENT.md`, or source files.

## Resource Constraint Policy

The primary server has only about `3 GB` of free RAM.

When making architecture or implementation decisions:

- prefer lower-memory options when the quality tradeoff is small
- avoid heavyweight defaults unless they provide a clear, material benefit
- if two viable options are close in quality but differ meaningfully in resource cost, present the lower-resource option first
- if a more expensive option is still worth considering, explain the tradeoff and let the user choose

## Deployment Policy

The new system must be isolated from the already-running website on the same server.

- Use Docker and `docker-compose.yml`.
- Do not break the existing Cloudflare Tunnel setup.
- Be explicit about routing boundaries between the old site and the new solar dashboard.
- Prefer reversible, low-risk networking changes.

## Data Policy

Preserve raw telemetry whenever possible. Filtered values are useful for UI and heuristics, but raw values are the canonical source for storage, analysis, and future ML work.

For microcontroller firmware:

- keep the MQTT contract server-compatible whenever possible
- avoid high-frequency flash writes that would wear out the ESP32 storage
- keep Wi-Fi and MQTT secrets in local untracked configuration such as `secrets.h`

## Documentation Hygiene

- Keep documentation implementation-ready.
- Record assumptions and decisions clearly.
- When requirements conflict, document the chosen resolution explicitly.
- Avoid leaking secrets or machine-specific private details into tracked files.
