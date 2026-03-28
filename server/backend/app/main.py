from __future__ import annotations

import asyncio
import json
import math
import os
import time as time_module
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from influxdb_client import InfluxDBClient


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class Settings:
    api_title: str = os.getenv("API_TITLE", "Sollar Panel API")
    influxdb_url: str = os.getenv("INFLUXDB_URL", "http://influxdb:8086")
    influxdb_token: str = os.getenv("INFLUXDB_TOKEN", "")
    influxdb_org: str = os.getenv("INFLUXDB_ORG", "sollar_panel")
    influxdb_bucket: str = os.getenv("INFLUXDB_BUCKET", "solar_metrics")
    influxdb_measurement: str = os.getenv("INFLUXDB_MEASUREMENT", "solar_voltage")
    default_sensor_id: str = os.getenv("INFLUXDB_SENSOR_ID", "pringles_1")
    timezone_name: str = os.getenv("SOLAR_TIMEZONE", "Europe/Copenhagen")
    model_registry_dir: str = os.getenv("MODEL_REGISTRY_DIR", "/models")
    live_poll_interval_seconds: float = float(os.getenv("LIVE_POLL_INTERVAL_SECONDS", "1.0"))
    analytics_cache_ttl_seconds: float = float(os.getenv("ANALYTICS_CACHE_TTL_SECONDS", "3.0"))
    history_cache_ttl_seconds: float = float(os.getenv("HISTORY_CACHE_TTL_SECONDS", "30.0"))


SETTINGS = Settings()
LOCAL_TIMEZONE = ZoneInfo(SETTINGS.timezone_name)


class InfluxRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
            timeout=10_000,
        )
        self.query_api = self.client.query_api()

    def close(self) -> None:
        self.client.close()

    def _query_points(
        self,
        sensor_id: str,
        start: datetime,
        stop: datetime | None = None,
        every_minutes: int | None = None,
    ) -> list[dict[str, Any]]:
        start_iso = start.astimezone(UTC).isoformat().replace("+00:00", "Z")
        stop_fragment = ""
        if stop is not None:
            stop_iso = stop.astimezone(UTC).isoformat().replace("+00:00", "Z")
            stop_fragment = f', stop: time(v: "{stop_iso}")'

        aggregate_fragment = ""
        if every_minutes is not None:
            aggregate_fragment = f"\n  |> aggregateWindow(every: {every_minutes}m, fn: mean, createEmpty: false)"

        query = f"""
from(bucket: "{self.settings.influxdb_bucket}")
  |> range(start: time(v: "{start_iso}"){stop_fragment})
  |> filter(fn: (r) => r["_measurement"] == "{self.settings.influxdb_measurement}")
  |> filter(fn: (r) => r["sensor_id"] == "{sensor_id}"){aggregate_fragment}
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
      "sample_count_5s",
      "uptime_seconds",
      "adc_raw",
      "temperature_c",
      "humidity_pct"
    ]
  )
  |> sort(columns: ["_time"])
"""
        tables = self.query_api.query(query)
        rows: list[dict[str, Any]] = []
        for table in tables:
            rows.extend(self._row_to_point(record) for record in table.records)
        return rows

    @staticmethod
    def _row_to_point(record: Any) -> dict[str, Any]:
        recorded_at = record.get_time()
        recorded_at_utc = recorded_at.astimezone(UTC)
        recorded_at_local = recorded_at_utc.astimezone(LOCAL_TIMEZONE)

        def read_float(field_name: str) -> float | None:
            value = record.values.get(field_name)
            return float(value) if value is not None else None

        def read_int(field_name: str) -> int | None:
            value = record.values.get(field_name)
            return int(value) if value is not None else None

        return {
            "timestamp": recorded_at_utc.isoformat().replace("+00:00", "Z"),
            "timestamp_local": recorded_at_local.isoformat(),
            "sensor_id": record.values.get("sensor_id"),
            "raw_voltage": read_float("raw_voltage"),
            "raw_voltage_last": read_float("raw_voltage_last"),
            "smoothed_voltage": read_float("smoothed_voltage"),
            "smoothed_voltage_last": read_float("smoothed_voltage_last"),
            "raw_min_5s": read_float("raw_min_5s"),
            "raw_max_5s": read_float("raw_max_5s"),
            "raw_mean_5s": read_float("raw_mean_5s"),
            "sample_count_5s": read_int("sample_count_5s"),
            "uptime_seconds": read_int("uptime_seconds"),
            "adc_raw": read_int("adc_raw"),
            "temperature_c": read_float("temperature_c"),
            "humidity_pct": read_float("humidity_pct"),
        }

    def fetch_latest(self, sensor_id: str) -> dict[str, Any] | None:
        points = self._query_points(sensor_id, start=utc_now() - timedelta(minutes=2))
        if not points:
            return None
        return points[-1]

    def fetch_history(self, sensor_id: str, target_day: date, every_minutes: int) -> list[dict[str, Any]]:
        start_local = datetime.combine(target_day, time.min, tzinfo=LOCAL_TIMEZONE)
        stop_local = start_local + timedelta(days=1)
        return self._query_points(
            sensor_id=sensor_id,
            start=start_local.astimezone(UTC),
            stop=stop_local.astimezone(UTC),
            every_minutes=every_minutes,
        )

    def fetch_recent(self, sensor_id: str, lookback: timedelta) -> list[dict[str, Any]]:
        return self._query_points(sensor_id, start=utc_now() - lookback)


app = FastAPI(title=SETTINGS.api_title, version="0.4.0")
app.state.settings = SETTINGS
app.state.influx = None
app.state.analytics_cache = {}
app.state.history_cache = {}


@app.on_event("startup")
def startup_event() -> None:
    app.state.influx = InfluxRepository(SETTINGS)


@app.on_event("shutdown")
def shutdown_event() -> None:
    repository = app.state.influx
    if repository is not None:
        repository.close()


def get_repository(request: Request) -> InfluxRepository:
    repository = request.app.state.influx
    if repository is None:
        raise HTTPException(status_code=503, detail="InfluxDB repository is not ready")
    return repository


def get_cached_payload(cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]], key: tuple[Any, ...]) -> dict[str, Any] | None:
    cached = cache.get(key)
    if cached is None:
        return None
    expires_at, payload = cached
    if expires_at <= time_module.monotonic():
        cache.pop(key, None)
        return None
    return payload


def put_cached_payload(
    cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]],
    key: tuple[Any, ...],
    ttl_seconds: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    cache[key] = (time_module.monotonic() + ttl_seconds, payload)
    return payload


def parse_sensor_id(sensor_id: str | None) -> str:
    value = sensor_id or SETTINGS.default_sensor_id
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if not set(value) <= allowed:
        raise HTTPException(status_code=400, detail="sensor_id contains unsupported characters")
    return value


def parse_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def local_now() -> datetime:
    return datetime.now(LOCAL_TIMEZONE)


def read_latest_insights() -> dict[str, Any] | None:
    insights_path = Path(SETTINGS.model_registry_dir) / "insights" / "latest.json"
    if not insights_path.exists():
        return None
    insights = json.loads(insights_path.read_text(encoding="utf-8"))
    insights["available"] = True
    return insights


def read_insights_history(limit: int = 96) -> list[dict[str, Any]]:
    history_path = Path(SETTINGS.model_registry_dir) / "insights" / "history.jsonl"
    if not history_path.exists():
        return []

    with history_path.open("r", encoding="utf-8") as history_file:
        rows = [json.loads(line) for line in history_file if line.strip()]
    return rows[-limit:]


def effective_voltage(point: dict[str, Any] | None) -> float | None:
    if point is None:
        return None
    for field_name in ("smoothed_voltage_last", "smoothed_voltage", "raw_mean_5s", "raw_voltage_last", "raw_voltage"):
        value = point.get(field_name)
        if value is not None:
            return float(value)
    return None


def raw_signal_value(point: dict[str, Any] | None) -> float | None:
    if point is None:
        return None
    for field_name in ("raw_voltage_last", "raw_voltage", "raw_mean_5s"):
        value = point.get(field_name)
        if value is not None:
            return float(value)
    return None


def classify_status(voltage: float | None) -> str:
    if voltage is None:
        return "No data"
    if voltage >= 0.34:
        return "Sunny"
    if voltage >= 0.2:
        return "Cloudy"
    if voltage >= 0.08:
        return "Twilight"
    return "Shade"


def format_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * ratio
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[int(position)])

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = position - lower_index
    return float(lower_value + (upper_value - lower_value) * fraction)


def nearest_historical_voltage(points: list[dict[str, Any]], target_time: datetime) -> float | None:
    for point in reversed(points):
        if parse_iso_timestamp(point["timestamp"]) <= target_time:
            return effective_voltage(point)
    return None


def build_delta_series(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not points:
        return []

    delta_series: list[dict[str, Any]] = []
    previous_point: dict[str, Any] | None = None
    for point in points:
        delta_value = 0.0
        current_voltage = raw_signal_value(point)
        if previous_point is not None and current_voltage is not None:
            previous_voltage = raw_signal_value(previous_point)
            if previous_voltage is not None:
                delta_seconds = (
                    parse_iso_timestamp(point["timestamp"]) - parse_iso_timestamp(previous_point["timestamp"])
                ).total_seconds()
                if delta_seconds > 0:
                    delta_value = (current_voltage - previous_voltage) / delta_seconds

        delta_series.append(
            {
                "timestamp": point["timestamp"],
                "timestamp_local": point["timestamp_local"],
                "delta_v_per_second": round(delta_value, 6),
            }
        )
        previous_point = point
    return delta_series


def build_feature_snapshot(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not points:
        return None

    latest = points[-1]
    latest_time = parse_iso_timestamp(latest["timestamp"])
    latest_voltage = effective_voltage(latest)
    if latest_voltage is None:
        return None

    one_minute_points = [
        effective_voltage(point)
        for point in points
        if parse_iso_timestamp(point["timestamp"]) >= latest_time - timedelta(minutes=1)
        and effective_voltage(point) is not None
    ]
    one_minute_std = None
    if len(one_minute_points) > 1:
        mean_value = sum(one_minute_points) / len(one_minute_points)
        variance = sum((value - mean_value) ** 2 for value in one_minute_points) / len(one_minute_points)
        one_minute_std = math.sqrt(variance)

    daily_points = [
        effective_voltage(point)
        for point in points
        if parse_iso_timestamp(point["timestamp"]).astimezone(LOCAL_TIMEZONE).date()
        == latest_time.astimezone(LOCAL_TIMEZONE).date()
        and effective_voltage(point) is not None
    ]
    daily_max = max(daily_points) if daily_points else None

    return {
        "rolling_std_1min": round(one_minute_std, 6) if one_minute_std is not None else None,
        "delta_v_5s": round(
            latest_voltage - (nearest_historical_voltage(points, latest_time - timedelta(seconds=5)) or latest_voltage),
            6,
        ),
        "delta_v_30s": round(
            latest_voltage - (nearest_historical_voltage(points, latest_time - timedelta(seconds=30)) or latest_voltage),
            6,
        ),
        "delta_v_5min": round(
            latest_voltage - (nearest_historical_voltage(points, latest_time - timedelta(minutes=5)) or latest_voltage),
            6,
        ),
        "voltage_to_daily_max_ratio": (
            round(latest_voltage / daily_max, 6) if daily_max not in {None, 0} else None
        ),
    }


def build_stats(points: list[dict[str, Any]]) -> dict[str, Any]:
    effective_values = [effective_voltage(point) for point in points if effective_voltage(point) is not None]
    raw_values = [raw_signal_value(point) for point in points if raw_signal_value(point) is not None]
    smoothed_values = [effective_voltage(point) for point in points if effective_voltage(point) is not None]
    latest_uptime = next(
        (point["uptime_seconds"] for point in reversed(points) if point.get("uptime_seconds") is not None),
        None,
    )

    signal_rms = None
    noise_rms = None
    snr_db = None
    if raw_values and smoothed_values and len(raw_values) == len(smoothed_values):
        signal_rms = math.sqrt(sum(value * value for value in smoothed_values) / len(smoothed_values))
        noise_values = [raw - smooth for raw, smooth in zip(raw_values, smoothed_values)]
        noise_rms = math.sqrt(sum(value * value for value in noise_values) / len(noise_values))
        if noise_rms > 0:
            snr_db = round(20 * math.log10(signal_rms / noise_rms), 2)

    return {
        "p50_voltage": round(percentile(effective_values, 0.50), 4) if effective_values else None,
        "p95_voltage": round(percentile(effective_values, 0.95), 4) if effective_values else None,
        "p99_voltage": round(percentile(effective_values, 0.99), 4) if effective_values else None,
        "snr_db": snr_db,
        "uptime_seconds": latest_uptime,
        "sample_count_last_hour": len(points),
        "bias_minutes_last_hour": None,
        "bias_sample_count_last_hour": 0,
    }


def build_analytics_payload(
    recent_points: list[dict[str, Any]],
    latest_insights: dict[str, Any] | None,
) -> dict[str, Any]:
    stats = build_stats(recent_points)
    if recent_points:
        latest_time = parse_iso_timestamp(recent_points[-1]["timestamp"])
        window_start = latest_time - timedelta(seconds=60)
        live_window = [
            point for point in recent_points if parse_iso_timestamp(point["timestamp"]) >= window_start
        ]
    else:
        live_window = []

    residual_history = read_insights_history()
    residual_points = [
        {
            "timestamp": row["trained_at_utc"],
            "timestamp_local": row["trained_at_local"],
            "residual_minutes": row.get("residual_minutes"),
            "confidence_level": row.get("confidence_level"),
            "predicted_phase": row.get("predicted_phase"),
        }
        for row in residual_history
        if row.get("residual_minutes") is not None
    ]
    bias_window_start = utc_now() - timedelta(hours=1)
    recent_residuals = [
        float(row["residual_minutes"])
        for row in residual_points
        if parse_iso_timestamp(row["timestamp"]) >= bias_window_start
    ]
    if recent_residuals:
        stats["bias_minutes_last_hour"] = round(sum(recent_residuals) / len(recent_residuals), 2)
        stats["bias_sample_count_last_hour"] = len(recent_residuals)

    latest = recent_points[-1] if recent_points else None
    return {
        "timezone": SETTINGS.timezone_name,
        "stats": stats,
        "latest": latest,
        "condition": classify_status(effective_voltage(latest)) if latest else "No data",
        "volatility_window_seconds": 60,
        "volatility_points": live_window,
        "delta_points": build_delta_series(live_window),
        "ai_residuals": residual_points,
        "latest_features": (
            latest_insights.get("latest_features")
            if latest_insights and latest_insights.get("latest_features")
            else build_feature_snapshot(recent_points)
        ),
        "phase_prediction": (
            latest_insights.get("phase_classifier")
            if latest_insights and latest_insights.get("phase_classifier")
            else None
        ),
        "latest_insights_available": bool(latest_insights and latest_insights.get("available")),
    }


@app.get("/healthz")
def healthz(request: Request) -> dict[str, str]:
    repository = get_repository(request)
    if not repository.client.ping():
        raise HTTPException(status_code=503, detail="InfluxDB is unreachable")
    return {"status": "ok", "service": "backend"}


@app.get("/api/status")
async def api_status(request: Request, sensor_id: str | None = None) -> dict[str, Any]:
    repository = get_repository(request)
    parsed_sensor_id = parse_sensor_id(sensor_id)
    latest = await asyncio.to_thread(repository.fetch_latest, parsed_sensor_id)
    insights = read_latest_insights()
    return {
        "status": "ok",
        "sensor_id": parsed_sensor_id,
        "bucket": SETTINGS.influxdb_bucket,
        "org": SETTINGS.influxdb_org,
        "measurement": SETTINGS.influxdb_measurement,
        "timezone": SETTINGS.timezone_name,
        "latest": latest,
        "condition": classify_status(effective_voltage(latest)) if latest else "No data",
        "phase_prediction": insights.get("phase_classifier") if insights else None,
    }


@app.get("/api/history")
async def api_history(
    request: Request,
    sensor_id: str | None = None,
    day: date | None = None,
    interval_minutes: int = Query(default=5, ge=1, le=60),
) -> dict[str, Any]:
    repository = get_repository(request)
    parsed_sensor_id = parse_sensor_id(sensor_id)
    target_day = day or local_now().date()
    cache_key = (parsed_sensor_id, target_day.isoformat(), interval_minutes)
    history_cache = request.app.state.history_cache
    cached_payload = get_cached_payload(history_cache, cache_key)
    if cached_payload is not None:
        return cached_payload

    points = await asyncio.to_thread(repository.fetch_history, parsed_sensor_id, target_day, interval_minutes)
    latest = points[-1] if points else None

    payload = {
        "sensor_id": parsed_sensor_id,
        "day": target_day.isoformat(),
        "interval_minutes": interval_minutes,
        "timezone": SETTINGS.timezone_name,
        "points": points,
        "latest": latest,
        "condition": classify_status(effective_voltage(latest)) if latest else "No data",
    }
    return put_cached_payload(history_cache, cache_key, SETTINGS.history_cache_ttl_seconds, payload)


@app.get("/api/insights")
def api_insights() -> dict[str, Any]:
    insights = read_latest_insights()
    if insights is None:
        return {
            "available": False,
            "timezone": SETTINGS.timezone_name,
            "message": "ML insights are not ready yet",
        }
    return insights


@app.get("/api/analytics")
async def api_analytics(request: Request, sensor_id: str | None = None) -> dict[str, Any]:
    repository = get_repository(request)
    parsed_sensor_id = parse_sensor_id(sensor_id)
    analytics_cache = request.app.state.analytics_cache
    cache_key = (parsed_sensor_id,)
    cached_payload = get_cached_payload(analytics_cache, cache_key)
    if cached_payload is not None:
        return cached_payload

    recent_points = await asyncio.to_thread(repository.fetch_recent, parsed_sensor_id, timedelta(hours=1))
    latest_insights = read_latest_insights()
    analytics = build_analytics_payload(recent_points, latest_insights)
    analytics["sensor_id"] = parsed_sensor_id
    return put_cached_payload(analytics_cache, cache_key, SETTINGS.analytics_cache_ttl_seconds, analytics)


@app.get("/api/live")
async def api_live(request: Request, sensor_id: str | None = None) -> StreamingResponse:
    repository = get_repository(request)
    parsed_sensor_id = parse_sensor_id(sensor_id)

    async def event_stream() -> Any:
        last_timestamp: str | None = None
        yield ": stream-open\n\n"
        yield format_sse(
            "status",
            {
                "state": "connected",
                "sensor_id": parsed_sensor_id,
                "at": utc_now().isoformat().replace("+00:00", "Z"),
                "at_local": local_now().isoformat(),
            },
        )

        while True:
            if await request.is_disconnected():
                break

            try:
                latest = await asyncio.to_thread(repository.fetch_latest, parsed_sensor_id)
            except Exception as exc:
                yield format_sse(
                    "stream-error",
                    {
                        "message": "Failed to fetch live telemetry",
                        "details": str(exc),
                    },
                )
                await asyncio.sleep(3)
                continue

            if latest is not None and latest["timestamp"] != last_timestamp:
                last_timestamp = latest["timestamp"]
                latest["condition"] = classify_status(effective_voltage(latest))
                insights = read_latest_insights()
                if insights is not None:
                    latest["predicted_phase"] = insights.get("phase_classifier", {}).get("predicted_phase")
                    latest["phase_confidence"] = insights.get("phase_classifier", {}).get("confidence")
                yield format_sse("telemetry", latest)
            else:
                yield format_sse(
                    "heartbeat",
                    {"at": utc_now().isoformat().replace("+00:00", "Z")},
                )

            await asyncio.sleep(SETTINGS.live_poll_interval_seconds)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
