from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("solar.ingestor")


@dataclass(slots=True)
class Settings:
    mqtt_host: str = os.getenv("MQTT_HOST", "mosquitto")
    mqtt_port: int = int(os.getenv("MQTT_PORT", "1883"))
    mqtt_topic: str = os.getenv("MQTT_TOPIC", "sensor/solar/voltage")
    mqtt_username: str = os.getenv("MQTT_USERNAME", "")
    mqtt_password: str = os.getenv("MQTT_PASSWORD", "")
    mqtt_client_id: str = os.getenv("MQTT_CLIENT_ID", "solar-ingestor")
    influxdb_url: str = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
    influxdb_token: str = os.getenv("INFLUXDB_TOKEN", "")
    influxdb_org: str = os.getenv("INFLUXDB_ORG", "sollar_panel")
    influxdb_bucket: str = os.getenv("INFLUXDB_BUCKET", "solar_metrics")
    influxdb_measurement: str = os.getenv("INFLUXDB_MEASUREMENT", "solar_voltage")
    default_sensor_id: str = os.getenv("INFLUXDB_SENSOR_ID", "esp32-solar-node")


def parse_timestamp(raw_value: str | None) -> datetime:
    if not raw_value:
        return datetime.now(UTC)

    normalized = raw_value.replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        LOGGER.warning("Invalid timestamp '%s'; using current UTC time instead", raw_value)
        return datetime.now(UTC)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)

    return parsed.astimezone(UTC)


class InfluxWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: InfluxDBClient | None = None
        self._write_api = None

    def ensure_ready(self) -> None:
        if self._client is not None and self._client.ping():
            return

        self.close()
        LOGGER.info("Connecting to InfluxDB at %s", self.settings.influxdb_url)
        self._client = InfluxDBClient(
            url=self.settings.influxdb_url,
            token=self.settings.influxdb_token,
            org=self.settings.influxdb_org,
            timeout=10_000,
        )
        if not self._client.ping():
            raise RuntimeError("InfluxDB is not ready yet")
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)

    def write(self, payload: dict[str, Any]) -> None:
        self.ensure_ready()

        raw_voltage = payload.get("raw_voltage", payload.get("raw_voltage_last"))
        if raw_voltage is None:
            raise ValueError("Payload is missing required field 'raw_voltage'")

        sensor_id = str(payload.get("sensor_id") or self.settings.default_sensor_id)
        timestamp = parse_timestamp(payload.get("timestamp"))
        smoothed_voltage = payload.get("smoothed_voltage", payload.get("smoothed_voltage_last"))
        min_v = payload.get("min_v", payload.get("raw_min_5s"))
        max_v = payload.get("max_v", payload.get("raw_max_5s"))
        mean_v = payload.get("mean_v", payload.get("raw_mean_5s"))
        sample_count = payload.get("sample_count", payload.get("sample_count_5s"))
        adc_raw = payload.get("adc_raw")
        temperature_c = payload.get("temperature_c")
        humidity_pct = payload.get("humidity_pct")

        point = (
            Point(self.settings.influxdb_measurement)
            .tag("sensor_id", sensor_id)
            .field("raw_voltage", float(raw_voltage))
            .field("raw_voltage_last", float(raw_voltage))
            .time(timestamp, WritePrecision.NS)
        )

        if smoothed_voltage is not None:
            point.field("smoothed_voltage", float(smoothed_voltage))
            point.field("smoothed_voltage_last", float(smoothed_voltage))

        if min_v is not None:
            point.field("min_v", float(min_v))
            point.field("raw_min_5s", float(min_v))
        if max_v is not None:
            point.field("max_v", float(max_v))
            point.field("raw_max_5s", float(max_v))
        if mean_v is not None:
            point.field("mean_v", float(mean_v))
            point.field("raw_mean_5s", float(mean_v))

        if sample_count is not None:
            point.field("sample_count", int(sample_count))
            point.field("sample_count_5s", int(sample_count))

        if adc_raw is not None:
            point.field("adc_raw", int(adc_raw))
        if temperature_c is not None:
            point.field("temperature_c", float(temperature_c))
        if humidity_pct is not None:
            point.field("humidity_pct", float(humidity_pct))

        uptime_seconds = payload.get("uptime_seconds")
        if uptime_seconds is not None:
            point.field("uptime_seconds", int(uptime_seconds))

        self._write_api.write(
            bucket=self.settings.influxdb_bucket,
            org=self.settings.influxdb_org,
            record=point,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._write_api = None


class SolarIngestionWorker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.influx_writer = InfluxWriter(settings)
        self.client = self._create_mqtt_client()

    def _create_mqtt_client(self) -> mqtt.Client:
        callback_api_version = getattr(mqtt, "CallbackAPIVersion", None)
        if callback_api_version is not None:
            client = mqtt.Client(
                callback_api_version.VERSION2,
                client_id=self.settings.mqtt_client_id,
                clean_session=False,
            )
        else:
            client = mqtt.Client(client_id=self.settings.mqtt_client_id, clean_session=False)

        if self.settings.mqtt_username:
            client.username_pw_set(
                username=self.settings.mqtt_username,
                password=self.settings.mqtt_password or None,
            )

        client.enable_logger(LOGGER)
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.on_connect = self.on_connect
        client.on_disconnect = self.on_disconnect
        client.on_message = self.on_message
        return client

    def wait_for_influxdb(self) -> None:
        while True:
            try:
                self.influx_writer.ensure_ready()
                LOGGER.info("InfluxDB connection established")
                return
            except Exception as exc:
                LOGGER.warning("InfluxDB not ready yet: %s", exc)
                time.sleep(5)

    def on_connect(
        self,
        client: mqtt.Client,
        _userdata: Any,
        _flags: Any,
        reason_code: Any,
        _properties: Any = None,
    ) -> None:
        if getattr(reason_code, "is_failure", False):
            LOGGER.error("MQTT connection failed with reason code %s", reason_code)
            return

        LOGGER.info("Connected to MQTT broker at %s:%s", self.settings.mqtt_host, self.settings.mqtt_port)
        client.subscribe(self.settings.mqtt_topic, qos=1)
        LOGGER.info("Subscribed to topic %s", self.settings.mqtt_topic)

    def on_disconnect(
        self,
        _client: mqtt.Client,
        _userdata: Any,
        reason_code: Any,
        _properties: Any = None,
    ) -> None:
        LOGGER.warning("Disconnected from MQTT broker with reason code %s", reason_code)

    def on_message(self, client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Skipping malformed JSON payload: %r", message.payload)
            return

        try:
            self.influx_writer.write(payload)
        except ValueError as exc:
            LOGGER.warning("Skipping invalid telemetry payload: %s; payload=%s", exc, payload)
            return
        except Exception as exc:
            LOGGER.exception("Failed to persist payload to InfluxDB: %s", exc)
            self.influx_writer.close()
            client.disconnect()
            return

        LOGGER.info(
            "Stored telemetry point: sensor_id=%s raw=%s smoothed=%s min=%s max=%s samples=%s",
            payload.get("sensor_id") or self.settings.default_sensor_id,
            payload.get("raw_voltage", payload.get("raw_voltage_last")),
            payload.get("smoothed_voltage", payload.get("smoothed_voltage_last")),
            payload.get("min_v", payload.get("raw_min_5s")),
            payload.get("max_v", payload.get("raw_max_5s")),
            payload.get("sample_count", payload.get("sample_count_5s")),
        )

    def run(self) -> None:
        self.wait_for_influxdb()
        LOGGER.info("Connecting to MQTT broker at %s:%s", self.settings.mqtt_host, self.settings.mqtt_port)
        self.client.connect(self.settings.mqtt_host, self.settings.mqtt_port, keepalive=60)
        self.client.loop_forever()


def main() -> None:
    settings = Settings()

    while True:
        worker = SolarIngestionWorker(settings)
        try:
            worker.run()
        except KeyboardInterrupt:
            LOGGER.info("Shutting down ingestion worker")
            worker.influx_writer.close()
            return
        except Exception as exc:
            LOGGER.exception("Worker crashed and will restart: %s", exc)
            worker.influx_writer.close()
            time.sleep(5)


if __name__ == "__main__":
    main()
