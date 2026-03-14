from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("solar.ml")


FEATURE_COLUMNS = [
    "effective_voltage",
    "hour",
    "minute",
    "volt_velocity",
    "rolling_mean_5min",
]


@dataclass(slots=True)
class Settings:
    influxdb_url: str = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
    influxdb_token: str = os.getenv("INFLUXDB_TOKEN", "")
    influxdb_org: str = os.getenv("INFLUXDB_ORG", "sollar_panel")
    influxdb_bucket: str = os.getenv("INFLUXDB_BUCKET", "solar_metrics")
    influxdb_measurement: str = os.getenv("INFLUXDB_MEASUREMENT", "solar_voltage")
    sensor_id: str = os.getenv("INFLUXDB_SENSOR_ID", "pringles_1")
    model_registry_dir: str = os.getenv("MODEL_REGISTRY_DIR", "/models")
    train_interval_minutes: int = int(os.getenv("ML_TRAIN_INTERVAL_MINUTES", "15"))
    lookback_days: int = int(os.getenv("ML_LOOKBACK_DAYS", "14"))
    timezone_name: str = os.getenv("SOLAR_TIMEZONE", "Europe/Copenhagen")
    night_threshold: float = float(os.getenv("ML_NIGHT_THRESHOLD", "0.01"))


class MlEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.timezone = ZoneInfo(settings.timezone_name)
        self.model_dir = Path(settings.model_registry_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.client = InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
            timeout=30_000,
        )
        self.query_api = self.client.query_api()

    def close(self) -> None:
        self.client.close()

    def query_recent_points(self) -> pd.DataFrame:
        query = f"""
from(bucket: "{self.settings.influxdb_bucket}")
  |> range(start: -{self.settings.lookback_days}d)
  |> filter(fn: (r) => r["_measurement"] == "{self.settings.influxdb_measurement}")
  |> filter(fn: (r) => r["sensor_id"] == "{self.settings.sensor_id}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "sensor_id", "raw_voltage", "smoothed_voltage"])
  |> sort(columns: ["_time"])
"""
        frames = self.query_api.query_data_frame(query)

        if isinstance(frames, list):
            data_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            data_frame = frames

        if data_frame.empty:
            return pd.DataFrame()

        data_frame = data_frame.rename(columns={"_time": "timestamp_utc"})
        return data_frame[["timestamp_utc", "sensor_id", "raw_voltage", "smoothed_voltage"]].copy()

    def preprocess(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        prepared = data_frame.copy()
        prepared["timestamp_utc"] = pd.to_datetime(prepared["timestamp_utc"], utc=True)
        prepared["timestamp_local"] = prepared["timestamp_utc"].dt.tz_convert(self.timezone)
        prepared["effective_voltage"] = prepared["smoothed_voltage"].fillna(prepared["raw_voltage"])
        prepared = prepared.dropna(subset=["effective_voltage"]).sort_values("timestamp_utc")

        prepared = prepared.set_index("timestamp_local")
        delta_seconds = prepared["timestamp_utc"].diff().dt.total_seconds().replace(0, np.nan)
        prepared["volt_velocity"] = prepared["effective_voltage"].diff().div(delta_seconds)
        prepared["rolling_mean_5min"] = prepared["effective_voltage"].rolling("5min", min_periods=1).mean()
        prepared["hour"] = prepared.index.hour
        prepared["minute"] = prepared.index.minute
        prepared["minute_of_day"] = prepared["hour"] * 60 + prepared["minute"] + (prepared.index.second / 60)
        prepared["volt_velocity"] = prepared["volt_velocity"].fillna(0.0)
        prepared["rolling_mean_5min"] = prepared["rolling_mean_5min"].fillna(prepared["effective_voltage"])
        prepared = prepared.reset_index()

        sunset_targets, sunrise_targets = self._compute_transition_targets(prepared)
        prepared["time_to_sunset_minutes"] = sunset_targets
        prepared["time_to_sunrise_minutes"] = sunrise_targets
        return prepared

    def _compute_transition_targets(self, prepared: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        timestamps = prepared["timestamp_utc"].astype("int64").to_numpy()
        effective = prepared["effective_voltage"].to_numpy()
        is_day = effective > self.settings.night_threshold

        sunset_mask = np.zeros_like(is_day, dtype=bool)
        sunrise_mask = np.zeros_like(is_day, dtype=bool)
        if len(is_day) > 1:
            sunset_mask[1:] = is_day[:-1] & (~is_day[1:])
            sunrise_mask[1:] = (~is_day[:-1]) & is_day[1:]

        sunset_times = timestamps[sunset_mask]
        sunrise_times = timestamps[sunrise_mask]

        sunset_targets = self._next_event_minutes(timestamps, sunset_times)
        sunrise_targets = self._next_event_minutes(timestamps, sunrise_times)
        return sunset_targets, sunrise_targets

    @staticmethod
    def _next_event_minutes(timestamps: np.ndarray, event_times: np.ndarray) -> np.ndarray:
        if event_times.size == 0:
            return np.full(shape=timestamps.shape, fill_value=np.nan, dtype=float)

        indexes = np.searchsorted(event_times, timestamps, side="left")
        valid = indexes < event_times.size
        next_event = np.full(shape=timestamps.shape, fill_value=np.nan, dtype=float)
        next_event[valid] = event_times[indexes[valid]]

        delta_minutes = (next_event - timestamps) / 60_000_000_000
        delta_minutes[delta_minutes <= 0] = np.nan
        delta_minutes[delta_minutes > 1_440] = np.nan
        return delta_minutes

    def train_linear_model(self, prepared: pd.DataFrame, target_column: str, model_name: str) -> dict[str, Any]:
        trainable = prepared.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()
        if len(trainable) < 40:
            return {
                "available": False,
                "model_name": model_name,
                "reason": "Not enough samples for training",
                "sample_count": int(len(trainable)),
            }

        split_index = max(int(len(trainable) * 0.8), 20)
        if len(trainable) - split_index < 10:
            split_index = len(trainable) - 10

        train_frame = trainable.iloc[:split_index]
        test_frame = trainable.iloc[split_index:]

        model = LinearRegression()
        model.fit(train_frame[FEATURE_COLUMNS], train_frame[target_column])

        predictions = model.predict(test_frame[FEATURE_COLUMNS])
        mae = float(mean_absolute_error(test_frame[target_column], predictions))
        r2 = float(r2_score(test_frame[target_column], predictions)) if len(test_frame) > 1 else 0.0

        model_path = self.model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        confidence = self._confidence_from_error(mae)
        return {
            "available": True,
            "model_name": model_name,
            "sample_count": int(len(trainable)),
            "test_sample_count": int(len(test_frame)),
            "mae_minutes": round(mae, 2),
            "r2_score": round(r2, 4),
            "confidence": confidence,
            "model_path": str(model_path),
            "model": model,
        }

    @staticmethod
    def _confidence_from_error(mae_minutes: float) -> str:
        if mae_minutes <= 20:
            return "High"
        if mae_minutes <= 60:
            return "Medium"
        return "Low"

    @staticmethod
    def _minutes_to_clock(predicted_minutes: float) -> str:
        normalized = max(0.0, min(1_439.0, predicted_minutes))
        total_minutes = int(round(normalized))
        hour = total_minutes // 60
        minute = total_minutes % 60
        return f"{hour:02d}:{minute:02d}"

    @staticmethod
    def _sanitize_eta(predicted_minutes: float | None) -> float | None:
        if predicted_minutes is None:
            return None
        if predicted_minutes < 0 or predicted_minutes > 1_440:
            return None
        return predicted_minutes

    @staticmethod
    def _minutes_to_eta(predicted_minutes: float | None) -> str:
        sanitized = MlEngine._sanitize_eta(predicted_minutes)
        if sanitized is None:
            return "Unavailable"
        return f"{int(round(sanitized))} min"

    def build_insights(self, prepared: pd.DataFrame) -> dict[str, Any]:
        latest = prepared.iloc[-1]

        time_model = self.train_linear_model(prepared, "minute_of_day", "time_of_day_model")
        sunset_model = self.train_linear_model(prepared, "time_to_sunset_minutes", "time_to_sunset_model")
        sunrise_model = self.train_linear_model(prepared, "time_to_sunrise_minutes", "time_to_sunrise_model")

        latest_features = latest[FEATURE_COLUMNS].to_frame().T
        ai_time_estimate = None
        sunset_eta = None
        sunrise_eta = None

        if time_model["available"]:
            ai_time_estimate = float(time_model["model"].predict(latest_features)[0])
        if sunset_model["available"]:
            sunset_eta = float(sunset_model["model"].predict(latest_features)[0])
        if sunrise_model["available"]:
            sunrise_eta = float(sunrise_model["model"].predict(latest_features)[0])

        sunset_eta = self._sanitize_eta(sunset_eta)
        sunrise_eta = self._sanitize_eta(sunrise_eta)

        available_confidences = [
            model["confidence"]
            for model in (time_model, sunset_model, sunrise_model)
            if model["available"]
        ]
        overall_confidence = available_confidences[0] if available_confidences else "Unavailable"
        if "Low" in available_confidences:
            overall_confidence = "Low"
        elif "Medium" in available_confidences:
            overall_confidence = "Medium"

        residual_minutes = None
        if ai_time_estimate is not None:
            residual_minutes = round(ai_time_estimate - float(latest["minute_of_day"]), 2)

        return {
            "timezone": self.settings.timezone_name,
            "sensor_id": self.settings.sensor_id,
            "trained_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "trained_at_local": datetime.now(self.timezone).isoformat(),
            "latest_point": {
                "timestamp": latest["timestamp_utc"].astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "timestamp_local": latest["timestamp_local"].isoformat(),
                "raw_voltage": round(float(latest["raw_voltage"]), 6) if pd.notna(latest["raw_voltage"]) else None,
                "smoothed_voltage": round(float(latest["smoothed_voltage"]), 6) if pd.notna(latest["smoothed_voltage"]) else None,
            },
            "ai_time_estimate": {
                "display_time": self._minutes_to_clock(ai_time_estimate) if ai_time_estimate is not None else "Unavailable",
                "predicted_minutes": round(ai_time_estimate, 2) if ai_time_estimate is not None else None,
                "confidence": time_model.get("confidence", "Unavailable"),
                "mae_minutes": time_model.get("mae_minutes"),
                "r2_score": time_model.get("r2_score"),
            },
            "estimated_sunset": {
                "display_eta": self._minutes_to_eta(sunset_eta),
                "minutes_until": round(sunset_eta, 2) if sunset_eta is not None else None,
                "confidence": sunset_model.get("confidence", "Unavailable"),
                "mae_minutes": sunset_model.get("mae_minutes"),
                "r2_score": sunset_model.get("r2_score"),
            },
            "estimated_sunrise": {
                "display_eta": self._minutes_to_eta(sunrise_eta),
                "minutes_until": round(sunrise_eta, 2) if sunrise_eta is not None else None,
                "confidence": sunrise_model.get("confidence", "Unavailable"),
                "mae_minutes": sunrise_model.get("mae_minutes"),
                "r2_score": sunrise_model.get("r2_score"),
            },
            "residual_minutes": residual_minutes,
            "confidence_level": overall_confidence,
        }

    def write_insights(self, insights: dict[str, Any]) -> None:
        models_dir = self.model_dir / "insights"
        models_dir.mkdir(parents=True, exist_ok=True)
        latest_path = models_dir / "latest.json"
        latest_path.write_text(json.dumps(insights, indent=2), encoding="utf-8")

        history_path = models_dir / "history.jsonl"
        history_entry = {
            "trained_at_utc": insights["trained_at_utc"],
            "trained_at_local": insights["trained_at_local"],
            "residual_minutes": insights.get("residual_minutes"),
            "confidence_level": insights.get("confidence_level"),
        }
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(history_entry) + "\n")

    def train_once(self) -> None:
        LOGGER.info("Starting online training cycle")
        data_frame = self.query_recent_points()
        if data_frame.empty:
            LOGGER.warning("No telemetry found in InfluxDB; skipping training cycle")
            return

        prepared = self.preprocess(data_frame)
        if prepared.empty:
            LOGGER.warning("Preprocessing produced no usable samples; skipping training cycle")
            return

        insights = self.build_insights(prepared)
        self.write_insights(insights)
        LOGGER.info(
            "Training cycle completed with confidence=%s latest_voltage=%s",
            insights["confidence_level"],
            insights["latest_point"]["smoothed_voltage"] or insights["latest_point"]["raw_voltage"],
        )


def main() -> None:
    settings = Settings()
    engine = MlEngine(settings)

    try:
        while True:
            try:
                engine.train_once()
            except Exception as exc:
                LOGGER.exception("Training cycle failed: %s", exc)
            time.sleep(settings.train_interval_minutes * 60)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
