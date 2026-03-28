# Project Overview

## Purpose

This project is an end-to-end IoT telemetry and machine learning preparation pipeline for a small solar panel node.

The short-term goal is reliable data collection, storage, visualization, and export.

The medium-term goal is to build a high-quality dataset that can support future models for sunlight condition classification and related forecasting or environmental inference tasks.

## System Context

The full system has two major parts:

- an edge collector running on an ESP32
- a server-side stack running on a Linux home server

The edge collector reads a low-voltage analog signal through the ESP32 ADC on `GPIO34`, reads `DHT11` temperature and humidity on `GPIO4`, keeps an OLED status display alive over I2C on `GPIO18`/`GPIO19`, smooths noisy readings, and publishes telemetry over MQTT.

The server receives telemetry, stores the raw signal in a time-series database, computes derived features, exposes historical and live data to a web UI, and exports daily training datasets for notebooks.

## Core Constraints

### Hardware and Signal Constraints

- The solar panel output is expected in the `0.0 V` to `0.5 V` range on the current hardware description.
- The active edge baseline uses the ESP32 ADC on `GPIO34`.
- The OLED display uses I2C address `0x3C` on `GPIO18` and `GPIO19`.
- A `DHT11` climate sensor is attached to `GPIO4`.
- The physical wiring is unstable and may produce very short disconnects.
- Low-light conditions produce significant noise and require smoothing.

### Server Constraints

- The primary server already hosts a public website through Cloudflare Tunnel.
- The new stack must be isolated and must not break the existing website.
- The primary server has only about `3 GB` of free RAM, so lightweight services are preferred.

## Recommended Architecture Summary

### Edge Side

- Arduino / PlatformIO acquisition loop
- `PubSubClient`
- `Adafruit SSD1306`
- `DHT sensor library`
- MQTT publish every second

### Server Side

- `Mosquitto` in Docker
- `InfluxDB` in Docker as the default database choice
- Python ingestion worker with reconnect handling for MQTT and DB writes
- Python feature and export job for daily dataset generation
- `FastAPI` backend for historical and live data
- dedicated frontend container for charts and status display

## Canonical Telemetry Contract

The requirements mention both smoothed publishing and raw-value storage. To avoid losing training-quality data, the recommended payload contract is:

- `timestamp`
- `adc_raw`
- `raw_voltage`
- `smoothed_voltage`
- `temperature_c`
- `humidity_pct`

This resolves the requirement conflict cleanly:

- `raw_voltage` remains the canonical series for storage and ML
- `smoothed_voltage` is available for UI display and simple heuristics
- climate fields are available for future feature engineering without changing the main solar contract

## Delivery Priorities

1. Build the resilient edge collector.
2. Stand up the Dockerized server ingestion path.
3. Store raw telemetry reliably.
4. Expose history and live data through the API.
5. Add the dashboard UI.
6. Export daily datasets for model training.
7. Replace heuristic status logic with an ML model later.
