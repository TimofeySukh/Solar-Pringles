# Edge Device Pipeline

## Target Hardware

- Device: Raspberry Pi Zero 2 W
- OS: Debian or Raspberry Pi OS
- ADC: ADS1115 on I2C address `0x48`
- Sensor input: small solar panel connected to ADS1115 `A0`

## Electrical and Reliability Constraints

- The current assembly is mechanically unstable and may produce sub-second disconnects.
- The software must tolerate intermittent read errors without crashing.
- Data loss during power loss should be minimized by flushing the local CSV file after each write.

## Required Python Stack

- Python 3
- `adafruit-circuitpython-ads1x15`
- `paho-mqtt`
- standard-library modules: `csv`, `time`, `datetime`

## Initialization Requirements

The I2C bus setup and ADS1115 initialization must use retry logic in an infinite loop until the device becomes available.

Required ADS1115 settings:

- initialize the channel with `AnalogIn(ads, 0)`
- do not use `ADS.P0`
- set `ads.gain = 4` to cover up to `1.024 V`

## Sampling Pipeline

### Frequency

- sample once per second

### Filtering

Use a simple moving average with a window size of `5` or `10` readings.

Implemented default:

- use `10` readings for stronger smoothing under noisy low-light conditions

### Read Error Handling

Every `chan.voltage` read must be wrapped in `try/except`.

The script must explicitly tolerate:

- `OSError: [Errno 121] Remote I/O error`
- `OSError: [Errno 5] Input/output error`

Required behavior on read failure:

- log the error
- sleep for `0.5` seconds
- continue the main loop
- do not terminate the process

## MQTT Publishing Requirements

Publish to:

- topic: `sensor/solar/voltage`

Recommended payload fields:

- `timestamp`
- `raw_voltage`
- `smoothed_voltage`
- `sensor_id`
- `uptime_seconds`

MQTT behavior requirements:

- publishing must not block the sampling loop indefinitely
- loss of broker connectivity or Wi-Fi must not crash the script
- use non-blocking publish or connection-safe exception handling

## Local Backup Requirements

Append every sampled row to `solar_log.csv` with these columns:

- `timestamp`
- `raw_voltage`
- `smoothed_voltage`

After each write:

- call `file.flush()`

This is mandatory to reduce buffered data loss during sudden power removal.

## Implementation Notes

The edge process should be treated as a long-running service with defensive behavior:

- retry hardware initialization forever
- never fail hard on transient ADC read errors
- continue sampling during MQTT outages
- keep a local CSV trail even when the network path is unavailable

## Implemented Node

The repository now includes a production-ready node implementation in `edge/solar_node.py`.

Behavior:

- initializes ADS1115 on address `0x48`
- uses `AnalogIn(ads, 0)`
- sets `ads.gain = 4`
- catches transient `OSError` values such as `Errno 5` and `Errno 121`
- skips the failed iteration instead of crashing
- computes a 10-sample simple moving average
- publishes to MQTT using `paho-mqtt`
- writes every successful sample to `solar_backup.csv`
- calls `flush()` after each backup row
- is deployable under `systemd` with `edge/systemd/sollar-panel-edge.service`

## Suggested Data Semantics

- `raw_voltage`: direct ADC-derived voltage before smoothing
- `smoothed_voltage`: moving average used for display and simple heuristics
- `timestamp`: UTC ISO 8601 timestamp generated at sample time
- `uptime_seconds`: Raspberry Pi uptime reported from `/proc/uptime` for dashboard operations telemetry
