from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("solar.ml")


PHASE_LABELS = ["Night", "Sunrise", "Day", "Sunset", "Anomaly"]
TIME_FEATURE_COLUMNS = [
    "raw_reference",
    "smoothed_reference",
    "rolling_std_1min",
    "delta_v_5s",
    "delta_v_30s",
    "delta_v_5min",
    "voltage_to_daily_max_ratio",
]
PHASE_FEATURE_COLUMNS = [
    "effective_voltage",
    "raw_reference",
    "smoothed_reference",
    "rolling_mean_5min",
    "rolling_std_1min",
    "delta_v_5s",
    "delta_v_30s",
    "delta_v_5min",
    "voltage_to_daily_max_ratio",
    "raw_window_range_5s",
    "raw_mean_5s",
    "sample_count_5s",
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
    training_stride_seconds: int = int(os.getenv("ML_TRAINING_STRIDE_SECONDS", "5"))
    timezone_name: str = os.getenv("SOLAR_TIMEZONE", "Europe/Copenhagen")
    night_threshold: float = float(os.getenv("ML_NIGHT_THRESHOLD", "0.035"))
    day_threshold: float = float(os.getenv("ML_DAY_THRESHOLD", "0.24"))
    anomaly_std_threshold: float = float(os.getenv("ML_ANOMALY_STD_THRESHOLD", "0.03"))
    anomaly_range_threshold: float = float(os.getenv("ML_ANOMALY_RANGE_THRESHOLD", "0.08"))
    sunrise_delta_threshold: float = float(os.getenv("ML_SUNRISE_DELTA_THRESHOLD", "0.01"))
    sunset_delta_threshold: float = float(os.getenv("ML_SUNSET_DELTA_THRESHOLD", "-0.01"))


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
  |> aggregateWindow(every: {self.settings.training_stride_seconds}s, fn: last, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(
    columns: [
      "_time",
      "sensor_id",
      "raw_voltage",
      "raw_voltage_last",
      "smoothed_voltage",
      "smoothed_voltage_last",
      "raw_min_5s",
      "raw_max_5s",
      "raw_mean_5s",
      "sample_count_5s"
    ]
  )
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
        expected_columns = [
            "timestamp_utc",
            "sensor_id",
            "raw_voltage",
            "raw_voltage_last",
            "smoothed_voltage",
            "smoothed_voltage_last",
            "raw_min_5s",
            "raw_max_5s",
            "raw_mean_5s",
            "sample_count_5s",
        ]
        for column in expected_columns:
            if column not in data_frame.columns:
                data_frame[column] = np.nan
        return data_frame[expected_columns].copy()

    def preprocess(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        prepared = data_frame.copy()
        prepared["timestamp_utc"] = pd.to_datetime(prepared["timestamp_utc"], utc=True)
        prepared = prepared.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")
        prepared["timestamp_local"] = prepared["timestamp_utc"].dt.tz_convert(self.timezone)

        prepared["raw_reference"] = prepared["raw_voltage_last"].fillna(prepared["raw_voltage"])
        prepared["smoothed_reference"] = prepared["smoothed_voltage_last"].fillna(prepared["smoothed_voltage"])
        prepared["effective_voltage"] = (
            prepared["smoothed_reference"]
            .fillna(prepared["raw_mean_5s"])
            .fillna(prepared["raw_reference"])
        )
        prepared = prepared.dropna(subset=["effective_voltage"]).reset_index(drop=True)
        if prepared.empty:
            return prepared

        prepared = prepared.set_index("timestamp_local")
        prepared["rolling_mean_5min"] = prepared["effective_voltage"].rolling("5min", min_periods=1).mean()
        prepared["rolling_std_1min"] = (
            prepared["effective_voltage"].rolling("1min", min_periods=2).std().fillna(0.0)
        )
        prepared["minute_of_day"] = (
            prepared.index.hour * 60
            + prepared.index.minute
            + (prepared.index.second / 60)
        )
        prepared["local_day"] = prepared.index.date
        prepared = prepared.reset_index()

        prepared["delta_v_5s"] = self._window_delta(prepared, window_seconds=5)
        prepared["delta_v_30s"] = self._window_delta(prepared, window_seconds=30)
        prepared["delta_v_5min"] = self._window_delta(prepared, window_seconds=300)
        prepared["sample_count_5s"] = prepared["sample_count_5s"].fillna(1).clip(lower=1)
        prepared["raw_mean_5s"] = prepared["raw_mean_5s"].fillna(prepared["raw_reference"])
        prepared["raw_min_5s"] = prepared["raw_min_5s"].fillna(prepared["raw_reference"])
        prepared["raw_max_5s"] = prepared["raw_max_5s"].fillna(prepared["raw_reference"])
        prepared["raw_window_range_5s"] = prepared["raw_max_5s"] - prepared["raw_min_5s"]

        running_daily_max = prepared.groupby("local_day")["effective_voltage"].cummax().clip(lower=1e-6)
        prepared["voltage_to_daily_max_ratio"] = (
            prepared["effective_voltage"] / running_daily_max
        ).clip(lower=0.0, upper=1.0)

        prepared["phase_label"] = prepared.apply(self._label_phase, axis=1)
        night_targets, _night_event_times = self._compute_event_targets(
            prepared,
            target_phase="Night",
            phase_gate={"Day", "Sunset"},
        )
        sunrise_phase_targets, _sunrise_event_times = self._compute_event_targets(
            prepared,
            target_phase="Sunrise",
            phase_gate={"Night", "Sunrise"},
        )
        prepared["time_to_sunset_minutes"] = night_targets
        prepared["time_to_sunrise_minutes"] = sunrise_phase_targets

        for column in set(TIME_FEATURE_COLUMNS + PHASE_FEATURE_COLUMNS):
            prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return prepared

    def _window_delta(self, prepared: pd.DataFrame, window_seconds: int) -> pd.Series:
        current = prepared[["timestamp_utc", "effective_voltage"]].copy()
        current["lookup_time"] = current["timestamp_utc"] - pd.to_timedelta(window_seconds, unit="s")
        historical = prepared[["timestamp_utc", "effective_voltage"]].copy().rename(
            columns={
                "timestamp_utc": "historical_time",
                "effective_voltage": "historical_voltage",
            }
        )
        merged = pd.merge_asof(
            current.sort_values("lookup_time"),
            historical.sort_values("historical_time"),
            left_on="lookup_time",
            right_on="historical_time",
            direction="backward",
        )
        delta = merged["effective_voltage"] - merged["historical_voltage"]
        return delta.fillna(0.0).reset_index(drop=True)

    def _label_phase(self, row: pd.Series) -> str:
        effective_voltage = float(row["effective_voltage"])
        ratio = float(row["voltage_to_daily_max_ratio"])
        rolling_std_1min = float(row["rolling_std_1min"])
        raw_window_range_5s = float(row["raw_window_range_5s"])
        delta_v_30s = float(row["delta_v_30s"])
        delta_v_5min = float(row["delta_v_5min"])

        if (
            rolling_std_1min >= self.settings.anomaly_std_threshold
            or raw_window_range_5s >= self.settings.anomaly_range_threshold
        ):
            return "Anomaly"
        if effective_voltage <= self.settings.night_threshold:
            return "Night"
        if delta_v_30s >= self.settings.sunrise_delta_threshold or (
            delta_v_5min > 0 and effective_voltage < self.settings.day_threshold
        ):
            return "Sunrise"
        if delta_v_30s <= self.settings.sunset_delta_threshold or (
            delta_v_5min < 0 and ratio < 0.9 and effective_voltage < self.settings.day_threshold
        ):
            return "Sunset"
        if effective_voltage >= self.settings.day_threshold or ratio >= 0.82:
            return "Day"
        return "Anomaly"

    @staticmethod
    def _entering_phase_mask(phases: pd.Series, target_phase: str) -> np.ndarray:
        current = phases.eq(target_phase).to_numpy()
        previous = phases.shift(1).eq(target_phase).fillna(False).to_numpy()
        return current & (~previous)

    def _compute_event_targets(
        self,
        prepared: pd.DataFrame,
        target_phase: str,
        phase_gate: set[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        timestamps = prepared["timestamp_utc"].astype("int64").to_numpy()
        phase_labels = prepared["phase_label"]
        event_times = timestamps[self._entering_phase_mask(phase_labels, target_phase)]
        delta_minutes = self._next_event_minutes(timestamps, event_times)
        gated_minutes = np.where(phase_labels.isin(phase_gate), delta_minutes, np.nan)
        return gated_minutes, event_times

    @staticmethod
    def _next_event_minutes(timestamps: np.ndarray, event_times: np.ndarray) -> np.ndarray:
        if event_times.size == 0:
            return np.full(shape=timestamps.shape, fill_value=np.nan, dtype=float)

        indexes = np.searchsorted(event_times, timestamps, side="left")
        valid = indexes < event_times.size
        next_event = np.full(shape=timestamps.shape, fill_value=np.nan, dtype=float)
        next_event[valid] = event_times[indexes[valid]]

        delta_minutes = (next_event - timestamps) / 60_000_000_000
        delta_minutes[delta_minutes < 0] = np.nan
        delta_minutes[delta_minutes > 1_440] = np.nan
        return delta_minutes

    def train_phase_classifier(self, prepared: pd.DataFrame) -> dict[str, Any]:
        trainable = prepared.dropna(subset=PHASE_FEATURE_COLUMNS + ["phase_label"]).copy()
        if len(trainable) < 80:
            return {
                "available": False,
                "reason": "Not enough samples for phase classifier training",
                "sample_count": int(len(trainable)),
            }

        split_index = max(int(len(trainable) * 0.8), 40)
        if len(trainable) - split_index < 20:
            split_index = len(trainable) - 20

        train_frame = trainable.iloc[:split_index]
        test_frame = trainable.iloc[split_index:]

        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        )
        model.fit(train_frame[PHASE_FEATURE_COLUMNS], train_frame["phase_label"])

        predictions = model.predict(test_frame[PHASE_FEATURE_COLUMNS])
        accuracy = float(accuracy_score(test_frame["phase_label"], predictions))
        model_path = self.model_dir / "phase_classifier.joblib"
        joblib.dump(model, model_path)

        return {
            "available": True,
            "sample_count": int(len(trainable)),
            "test_sample_count": int(len(test_frame)),
            "accuracy": round(accuracy, 4),
            "confidence": self._confidence_from_accuracy(accuracy),
            "model_path": str(model_path),
            "model": model,
        }

    def train_regressor(
        self,
        prepared: pd.DataFrame,
        target_column: str,
        model_name: str,
        allowed_phases: set[str],
        feature_columns: list[str],
    ) -> dict[str, Any]:
        phase_mask = prepared["phase_label"].isin(allowed_phases)
        trainable = prepared[phase_mask].dropna(subset=feature_columns + [target_column]).copy()
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
        model.fit(train_frame[feature_columns], train_frame[target_column])

        predictions = model.predict(test_frame[feature_columns])
        mae = float(mean_absolute_error(test_frame[target_column], predictions))
        r2 = float(r2_score(test_frame[target_column], predictions)) if len(test_frame) > 1 else 0.0

        model_path = self.model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        return {
            "available": True,
            "model_name": model_name,
            "sample_count": int(len(trainable)),
            "test_sample_count": int(len(test_frame)),
            "mae_minutes": round(mae, 2),
            "r2_score": round(r2, 4),
            "confidence": self._confidence_from_error(mae),
            "model_path": str(model_path),
            "model": model,
        }

    @staticmethod
    def _confidence_from_error(mae_minutes: float) -> str:
        if mae_minutes <= 15:
            return "High"
        if mae_minutes <= 45:
            return "Medium"
        return "Low"

    @staticmethod
    def _confidence_from_accuracy(accuracy: float) -> str:
        if accuracy >= 0.9:
            return "High"
        if accuracy >= 0.75:
            return "Medium"
        return "Low"

    @staticmethod
    def _minutes_to_clock(predicted_minutes: float | None) -> str:
        if predicted_minutes is None:
            return "Unavailable"
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

    @staticmethod
    def _worst_confidence(levels: list[str]) -> str:
        if not levels:
            return "Unavailable"
        if "Low" in levels:
            return "Low"
        if "Medium" in levels:
            return "Medium"
        return "High"

    def build_insights(self, prepared: pd.DataFrame) -> dict[str, Any]:
        latest = prepared.iloc[-1]
        latest_time_features = latest[TIME_FEATURE_COLUMNS].to_frame().T
        latest_phase_features = latest[PHASE_FEATURE_COLUMNS].to_frame().T

        time_model = self.train_regressor(
            prepared,
            target_column="minute_of_day",
            model_name="time_of_day_model",
            allowed_phases={"Night", "Sunrise", "Day", "Sunset"},
            feature_columns=TIME_FEATURE_COLUMNS,
        )
        phase_classifier = self.train_phase_classifier(prepared)
        sunset_model = self.train_regressor(
            prepared,
            target_column="time_to_sunset_minutes",
            model_name="time_to_sunset_model",
            allowed_phases={"Day", "Sunset"},
            feature_columns=TIME_FEATURE_COLUMNS,
        )
        sunrise_model = self.train_regressor(
            prepared,
            target_column="time_to_sunrise_minutes",
            model_name="time_to_sunrise_model",
            allowed_phases={"Night", "Sunrise"},
            feature_columns=TIME_FEATURE_COLUMNS,
        )

        ai_time_estimate = None
        predicted_phase = latest["phase_label"]
        phase_probability = None
        sunset_eta = None
        sunrise_eta = None

        if time_model["available"]:
            ai_time_estimate = float(time_model["model"].predict(latest_time_features)[0])
        if phase_classifier["available"]:
            predicted_phase = str(phase_classifier["model"].predict(latest_phase_features)[0])
            probabilities = phase_classifier["model"].predict_proba(latest_phase_features)[0]
            phase_probability = float(np.max(probabilities))
        if sunset_model["available"] and predicted_phase in {"Day", "Sunset"}:
            sunset_eta = float(sunset_model["model"].predict(latest_time_features)[0])
        if sunrise_model["available"] and predicted_phase in {"Night", "Sunrise"}:
            sunrise_eta = float(sunrise_model["model"].predict(latest_time_features)[0])

        sunset_eta = self._sanitize_eta(sunset_eta)
        sunrise_eta = self._sanitize_eta(sunrise_eta)
        residual_minutes = None
        if ai_time_estimate is not None:
            residual_minutes = round(ai_time_estimate - float(latest["minute_of_day"]), 2)

        confidence_levels = [
            model["confidence"]
            for model in (time_model, sunset_model, sunrise_model)
            if model.get("available")
        ]
        if phase_classifier.get("available"):
            confidence_levels.append(phase_classifier["confidence"])
        if predicted_phase == "Anomaly":
            confidence_levels.append("Low")

        return {
            "timezone": self.settings.timezone_name,
            "sensor_id": self.settings.sensor_id,
            "trained_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "trained_at_local": datetime.now(self.timezone).isoformat(),
            "latest_point": {
                "timestamp": latest["timestamp_utc"].astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "timestamp_local": latest["timestamp_local"].isoformat(),
                "raw_voltage": round(float(latest["raw_reference"]), 6) if pd.notna(latest["raw_reference"]) else None,
                "smoothed_voltage": round(float(latest["smoothed_reference"]), 6) if pd.notna(latest["smoothed_reference"]) else None,
            },
            "latest_features": {
                "raw_voltage": round(float(latest["raw_reference"]), 6) if pd.notna(latest["raw_reference"]) else None,
                "smoothed_voltage": (
                    round(float(latest["smoothed_reference"]), 6) if pd.notna(latest["smoothed_reference"]) else None
                ),
                "rolling_std_1min": round(float(latest["rolling_std_1min"]), 6),
                "delta_v_5s": round(float(latest["delta_v_5s"]), 6),
                "delta_v_30s": round(float(latest["delta_v_30s"]), 6),
                "delta_v_5min": round(float(latest["delta_v_5min"]), 6),
                "voltage_to_daily_max_ratio": round(float(latest["voltage_to_daily_max_ratio"]), 6),
                "raw_window_range_5s": round(float(latest["raw_window_range_5s"]), 6),
            },
            "phase_classifier": {
                "predicted_phase": predicted_phase,
                "confidence": phase_classifier.get("confidence", "Unavailable"),
                "accuracy": phase_classifier.get("accuracy"),
                "predicted_probability": round(phase_probability, 4) if phase_probability is not None else None,
            },
            "ai_time_estimate": {
                "display_time": self._minutes_to_clock(ai_time_estimate),
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
                "active_for_phase": predicted_phase in {"Day", "Sunset"},
            },
            "estimated_sunrise": {
                "display_eta": self._minutes_to_eta(sunrise_eta),
                "minutes_until": round(sunrise_eta, 2) if sunrise_eta is not None else None,
                "confidence": sunrise_model.get("confidence", "Unavailable"),
                "mae_minutes": sunrise_model.get("mae_minutes"),
                "r2_score": sunrise_model.get("r2_score"),
                "active_for_phase": predicted_phase in {"Night", "Sunrise"},
            },
            "residual_minutes": residual_minutes,
            "confidence_level": self._worst_confidence(confidence_levels),
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
            "predicted_phase": insights.get("phase_classifier", {}).get("predicted_phase"),
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
            "Training cycle completed with phase=%s confidence=%s latest_voltage=%s",
            insights["phase_classifier"]["predicted_phase"],
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
