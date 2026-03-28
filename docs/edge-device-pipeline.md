# Edge Device Pipeline

## Target Hardware

- Device baseline: ESP32
- Solar input: small solar panel connected to `GPIO34`
- Climate sensor: `DHT11` on `GPIO4`
- OLED display: SSD1306 at `0x3C`
- OLED I2C pins:
  - `SDA`: `GPIO18`
  - `SCL`: `GPIO19`

## Electrical and Reliability Constraints

- The current assembly is mechanically unstable and may produce sub-second disconnects.
- Wi-Fi availability can fluctuate and must not stall the sampling loop.
- The OLED display must keep refreshing even if network or DHT reads fail.
- Flash-heavy per-sample persistence is a poor fit for an ESP32 and should be avoided.

## Required ESP32 Stack

- PlatformIO
- Arduino framework for ESP32
- `PubSubClient`
- `Adafruit SSD1306`
- `Adafruit GFX`
- `DHT sensor library`
- `ArduinoJson`

## Initialization Requirements

The ESP32 edge firmware must initialize these subsystems independently:

- Wi-Fi
- NTP time sync
- MQTT
- OLED display
- DHT11
- ADC on `GPIO34`

Display initialization must be retry-safe so the screen can recover even if I2C glitches during boot.

## Sampling Pipeline

### Frequency

- sample five times per second
- publish one aggregate MQTT packet every second
- refresh the OLED every second
- read DHT11 no faster than every three seconds

### Filtering

Use a simple moving average with a window size of `5` or `10` readings.

Implemented default:

- use `10` readings for stronger smoothing under noisy low-light conditions
- keep local sampling at `5 Hz`
- publish one aggregate MQTT packet every `1 second`

### Read Error Handling

Required behavior on edge-side failures:

- ADC reads must not block Wi-Fi, MQTT, or display refresh.
- DHT11 read failures must keep the main loop alive.
- MQTT disconnects must trigger reconnect attempts without halting sampling.
- OLED failures must trigger periodic re-initialization attempts instead of a hard stop.

## MQTT Publishing Requirements

Publish to:

- topic: `sensor/solar/voltage`

Recommended payload fields:

- `timestamp`
- `adc_raw`
- `raw_voltage`
- `smoothed_voltage`
- `min_v`
- `max_v`
- `mean_v`
- `temperature_c`
- `humidity_pct`
- `sample_count`
- `sensor_id`
- `uptime_seconds`

MQTT behavior requirements:

- publishing must not block the sampling loop indefinitely
- loss of broker connectivity or Wi-Fi must not crash the script
- use non-blocking publish or connection-safe exception handling

## Local State Requirements

The ESP32 migration intentionally avoids per-sample persistent CSV logging on internal flash.

Reason:

- constant flash writes at `5 Hz` are not a good durability tradeoff on a microcontroller

The firmware should instead keep:

- a live smoothing window in RAM
- a one-second publish aggregate in RAM
- retry-safe Wi-Fi and MQTT reconnect behavior

## Implementation Notes

The edge firmware should be treated as a long-running live device:

- keep solar sampling independent from display refresh
- keep display refresh independent from DHT reads
- keep MQTT reconnect logic independent from sensor reads
- publish a backend-compatible payload so the server path does not fork

## Implemented Node

The repository now includes the ESP32 edge implementation in `edge/esp32/`.

Current ESP32 behavior:

- samples `GPIO34` at `5 Hz`
- computes a 10-sample simple moving average
- reads `DHT11` on `GPIO4`
- refreshes the OLED over `GPIO18`/`GPIO19`
- publishes MQTT aggregates every `1 second`
- sends `temperature_c`, `humidity_pct`, `adc_raw`, voltage stats, and `uptime_seconds`
- retries Wi-Fi, MQTT, and OLED initialization without blocking the whole device

## Suggested Data Semantics

- `raw_voltage`: most recent ADC-derived voltage inside the current 1-second publish window
- `smoothed_voltage`: most recent moving-average value inside the current 1-second publish window
- `adc_raw`: latest raw ESP32 ADC value from `GPIO34`
- `min_v`: minimum raw voltage seen inside the current 1-second publish window
- `max_v`: maximum raw voltage seen inside the current 1-second publish window
- `mean_v`: mean raw voltage across the current 1-second publish window
- `sample_count`: number of edge samples included in the current publish window, normally `5`
- `timestamp`: UTC ISO 8601 timestamp of the latest sample in the publish window
- `temperature_c`: latest valid DHT11 temperature reading in Celsius
- `humidity_pct`: latest valid DHT11 relative humidity reading
- `uptime_seconds`: ESP32 uptime in seconds for dashboard operations telemetry
