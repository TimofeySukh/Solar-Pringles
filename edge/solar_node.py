from __future__ import annotations

import csv
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any

import board
import busio
import paho.mqtt.client as mqtt
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn


logging.basicConfig(
    level=os.getenv("SOLAR_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("solar.node")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def system_uptime_seconds() -> int | None:
    try:
        uptime_text = Path("/proc/uptime").read_text(encoding="utf-8").split()[0]
        return int(float(uptime_text))
    except Exception as exc:
        LOGGER.warning("Failed to read system uptime: %s", exc)
        return None


@dataclass(slots=True)
class Sample:
    timestamp: str
    raw_voltage: float
    smoothed_voltage: float


@dataclass(slots=True)
class Settings:
    sensor_id: str = os.getenv("SOLAR_SENSOR_ID", "pringles_1")
    mqtt_host: str = os.getenv("SOLAR_MQTT_HOST", "")
    mqtt_port: int = int(os.getenv("SOLAR_MQTT_PORT", "1884"))
    mqtt_topic: str = os.getenv("SOLAR_MQTT_TOPIC", "sensor/solar/voltage")
    sample_interval_seconds: float = float(os.getenv("SOLAR_SAMPLE_INTERVAL_SECONDS", "0.2"))
    publish_interval_seconds: float = float(os.getenv("SOLAR_PUBLISH_INTERVAL_SECONDS", "1.0"))
    smoothing_window: int = int(os.getenv("SOLAR_SMOOTHING_WINDOW", "10"))
    backup_path: str = os.getenv("SOLAR_BACKUP_PATH", "/opt/sollar_panel/solar_backup.csv")
    ads_i2c_address: int = int(os.getenv("SOLAR_ADS_I2C_ADDRESS", "0x48"), 16)
    ads_gain: int = int(os.getenv("SOLAR_ADS_GAIN", "4"))

    def validate(self) -> None:
        if not self.mqtt_host:
            raise ValueError("SOLAR_MQTT_HOST must be configured")
        if self.smoothing_window < 1:
            raise ValueError("SOLAR_SMOOTHING_WINDOW must be at least 1")
        if self.sample_interval_seconds <= 0:
            raise ValueError("SOLAR_SAMPLE_INTERVAL_SECONDS must be positive")
        if self.publish_interval_seconds <= 0:
            raise ValueError("SOLAR_PUBLISH_INTERVAL_SECONDS must be positive")
        if self.publish_interval_seconds < self.sample_interval_seconds:
            raise ValueError("SOLAR_PUBLISH_INTERVAL_SECONDS must be >= SOLAR_SAMPLE_INTERVAL_SECONDS")


class BackupWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not file_exists:
            self._writer.writerow(["timestamp", "raw_voltage", "smoothed_voltage"])
            self._file.flush()

    def write_row(self, timestamp: str, raw_voltage: float, smoothed_voltage: float) -> None:
        self._writer.writerow(
            [
                timestamp,
                f"{raw_voltage:.6f}",
                f"{smoothed_voltage:.6f}",
            ]
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class SensorReader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._chan: AnalogIn | None = None
        self._samples: deque[float] = deque(maxlen=settings.smoothing_window)

    def _connect(self) -> None:
        while self._chan is None:
            try:
                LOGGER.info(
                    "Initializing ADS1115 on I2C address %s with gain %s",
                    hex(self.settings.ads_i2c_address),
                    self.settings.ads_gain,
                )
                i2c = busio.I2C(board.SCL, board.SDA)
                ads = ADS1115(i2c, address=self.settings.ads_i2c_address)
                ads.gain = self.settings.ads_gain
                self._chan = AnalogIn(ads, 0)
                LOGGER.info("ADS1115 ready on channel 0")
            except Exception as exc:
                LOGGER.warning("ADS1115 initialization failed: %s", exc)
                time.sleep(2)

    def read(self) -> Sample | None:
        if self._chan is None:
            self._connect()

        try:
            raw_voltage = float(self._chan.voltage)
        except OSError as exc:
            errno_value = getattr(exc, "errno", None)
            if errno_value in {5, 121}:
                LOGGER.warning("Transient I2C read error (%s): %s", errno_value, exc)
                return None
            LOGGER.exception("Unexpected I2C OSError")
            return None
        except Exception as exc:
            LOGGER.exception("Unexpected sensor read failure: %s", exc)
            self._chan = None
            return None

        self._samples.append(raw_voltage)
        smoothed_voltage = sum(self._samples) / len(self._samples)
        return Sample(
            timestamp=utc_now_iso(),
            raw_voltage=raw_voltage,
            smoothed_voltage=smoothed_voltage,
        )


class MqttPublisher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.connected = Event()
        self.stop_requested = Event()
        self._connect_lock = Lock()

        callback_api_version = getattr(mqtt, "CallbackAPIVersion", None)
        if callback_api_version is not None:
            self.client = mqtt.Client(
                callback_api_version.VERSION2,
                client_id=f"{settings.sensor_id}-publisher",
                clean_session=True,
            )
        else:
            self.client = mqtt.Client(
                client_id=f"{settings.sensor_id}-publisher",
                clean_session=True,
            )

        self.client.enable_logger(LOGGER)
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, _client, _userdata, _flags, reason_code, _properties=None) -> None:
        if getattr(reason_code, "is_failure", False):
            LOGGER.error("MQTT connection failed: %s", reason_code)
            self.connected.clear()
            return

        LOGGER.info("Connected to MQTT broker at %s:%s", self.settings.mqtt_host, self.settings.mqtt_port)
        self.connected.set()

    def on_disconnect(
        self,
        _client,
        _userdata,
        reason_code=None,
        _properties=None,
        *_extra,
    ) -> None:
        self.connected.clear()
        if self.stop_requested.is_set():
            LOGGER.info("MQTT client disconnected")
            return
        LOGGER.warning("MQTT disconnected with reason code %s; automatic reconnect is active", reason_code)

    def start(self) -> None:
        with self._connect_lock:
            LOGGER.info("Starting MQTT client")
            self.client.connect_async(self.settings.mqtt_host, self.settings.mqtt_port, keepalive=60)
            self.client.loop_start()

    def publish(self, payload: dict[str, object]) -> None:
        message = json.dumps(payload, separators=(",", ":"))
        result = self.client.publish(self.settings.mqtt_topic, message, qos=1)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            LOGGER.info(
                "Published golden-second telemetry: raw=%s smoothed=%s min=%s max=%s samples=%s",
                payload["raw_voltage"],
                payload["smoothed_voltage"],
                payload["min_v"],
                payload["max_v"],
                payload["sample_count"],
            )
            return

        LOGGER.warning("MQTT publish queued unsuccessfully with rc=%s; retaining local backup only", result.rc)

    def stop(self) -> None:
        self.stop_requested.set()
        self.client.loop_stop()
        self.client.disconnect()


class SolarNode:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stop_requested = Event()
        self.sensor_reader = SensorReader(settings)
        self.mqtt_publisher = MqttPublisher(settings)
        self.backup_writer = BackupWriter(settings.backup_path)
        self.publish_window: deque[Sample] = deque()
        self._last_publish_monotonic = time.monotonic()

    @staticmethod
    def _build_payload(settings: Settings, samples: list[Sample]) -> dict[str, object]:
        latest = samples[-1]
        raw_values = [sample.raw_voltage for sample in samples]
        payload: dict[str, object] = {
            "sensor_id": settings.sensor_id,
            "timestamp": latest.timestamp,
            "raw_voltage": round(latest.raw_voltage, 6),
            "smoothed_voltage": round(latest.smoothed_voltage, 6),
            "min_v": round(min(raw_values), 6),
            "max_v": round(max(raw_values), 6),
            "mean_v": round(sum(raw_values) / len(raw_values), 6),
            "sample_count": len(samples),
        }
        uptime_seconds = system_uptime_seconds()
        if uptime_seconds is not None:
            payload["uptime_seconds"] = uptime_seconds
        return payload

    def _drain_publish_window(self) -> dict[str, object] | None:
        if not self.publish_window:
            return None
        samples = list(self.publish_window)
        self.publish_window.clear()
        return self._build_payload(self.settings, samples)

    def stop(self, *_args) -> None:
        LOGGER.info("Stop requested")
        self.stop_requested.set()

    def run(self) -> None:
        self.mqtt_publisher.start()
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

        try:
            while not self.stop_requested.is_set():
                loop_started_at = time.monotonic()
                sample = self.sensor_reader.read()

                if sample is not None:
                    self.backup_writer.write_row(
                        sample.timestamp,
                        raw_voltage=sample.raw_voltage,
                        smoothed_voltage=sample.smoothed_voltage,
                    )
                    self.publish_window.append(sample)

                if loop_started_at - self._last_publish_monotonic >= self.settings.publish_interval_seconds:
                    payload = self._drain_publish_window()
                    self._last_publish_monotonic = loop_started_at
                    if payload is not None:
                        self.mqtt_publisher.publish(payload)

                elapsed = time.monotonic() - loop_started_at
                sleep_for = max(0.0, self.settings.sample_interval_seconds - elapsed)
                self.stop_requested.wait(sleep_for)
        finally:
            if self.publish_window:
                payload = self._drain_publish_window()
                if payload is not None:
                    self.mqtt_publisher.publish(payload)
            self.mqtt_publisher.stop()
            self.backup_writer.close()


def main() -> int:
    settings = Settings()
    try:
        settings.validate()
    except ValueError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 1

    node = SolarNode(settings)
    node.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
