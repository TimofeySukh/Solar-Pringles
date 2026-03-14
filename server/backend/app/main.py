from __future__ import annotations

import asyncio
import json
import math
import os
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
            aggregate_fragment = f'\n  |> aggregateWindow(every: {every_minutes}m, fn: mean, createEmpty: false)'

        query = f"""
from(bucket: "{self.settings.influxdb_bucket}")
  |> range(start: time(v: "{start_iso}"){stop_fragment})
  |> filter(fn: (r) => r["_measurement"] == "{self.settings.influxdb_measurement}")
  |> filter(fn: (r) => r["sensor_id"] == "{sensor_id}"){aggregate_fragment}
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "sensor_id", "raw_voltage", "smoothed_voltage", "uptime_seconds"])
  |> sort(columns: ["_time"])
"""
        tables = self.query_api.query(query)
        rows: list[dict[str, Any]] = []
        for table in tables:
            rows.extend(self._row_to_point(record) for record in table.records)
        return rows

    @staticmethod
    def _row_to_point(record: Any) -> dict[str, Any]:
        raw_voltage = record.values.get("raw_voltage")
        smoothed_voltage = record.values.get("smoothed_voltage")
        uptime_seconds = record.values.get("uptime_seconds")
        recorded_at = record.get_time()
        recorded_at_utc = recorded_at.astimezone(UTC)
        recorded_at_local = recorded_at_utc.astimezone(LOCAL_TIMEZONE)

        return {
            "timestamp": recorded_at_utc.isoformat().replace("+00:00", "Z"),
            "timestamp_local": recorded_at_local.isoformat(),
            "sensor_id": record.values.get("sensor_id"),
            "raw_voltage": float(raw_voltage) if raw_voltage is not None else None,
            "smoothed_voltage": float(smoothed_voltage) if smoothed_voltage is not None else None,
            "uptime_seconds": int(uptime_seconds) if uptime_seconds is not None else None,
        }

    def fetch_latest(self, sensor_id: str) -> dict[str, Any] | None:
        rows = self._query_points(sensor_id, start=utc_now() - timedelta(hours=12))
        return rows[-1] if rows else None

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


app = FastAPI(title=SETTINGS.api_title, version="0.3.0")
app.state.settings = SETTINGS
app.state.influx = None


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
    return json.loads(insights_path.read_text(encoding="utf-8"))


def read_insights_history(limit: int = 96) -> list[dict[str, Any]]:
    history_path = Path(SETTINGS.model_registry_dir) / "insights" / "history.jsonl"
    if not history_path.exists():
        return []

    with history_path.open("r", encoding="utf-8") as history_file:
        rows = [json.loads(line) for line in history_file if line.strip()]
    return rows[-limit:]


def effective_voltage(point: dict[str, Any]) -> float | None:
    return point.get("smoothed_voltage") if point.get("smoothed_voltage") is not None else point.get("raw_voltage")


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


def build_delta_series(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not points:
        return []

    delta_series: list[dict[str, Any]] = []
    previous_point: dict[str, Any] | None = None
    for point in points:
        delta_value = 0.0
        if previous_point is not None:
            current_voltage = point.get("raw_voltage")
            previous_voltage = previous_point.get("raw_voltage")
            if current_voltage is not None and previous_voltage is not None:
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


def build_stats(points: list[dict[str, Any]]) -> dict[str, Any]:
    effective_values = [effective_voltage(point) for point in points if effective_voltage(point) is not None]
    raw_values = [point["raw_voltage"] for point in points if point.get("raw_voltage") is not None]
    smoothed_values = [point["smoothed_voltage"] for point in points if point.get("smoothed_voltage") is not None]
    latest_uptime = next((point["uptime_seconds"] for point in reversed(points) if point.get("uptime_seconds") is not None), None)

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
    }


def build_analytics_payload(
    recent_points: list[dict[str, Any]],
    latest_insights: dict[str, Any] | None,
) -> dict[str, Any]:
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
        }
        for row in residual_history
        if row.get("residual_minutes") is not None
    ]

    latest = recent_points[-1] if recent_points else None
    return {
        "timezone": SETTINGS.timezone_name,
        "stats": build_stats(recent_points),
        "latest": latest,
        "condition": classify_status(effective_voltage(latest)) if latest else "No data",
        "volatility_window_seconds": 60,
        "volatility_points": live_window,
        "delta_points": build_delta_series(live_window),
        "ai_residuals": residual_points,
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
    return {
        "status": "ok",
        "sensor_id": parsed_sensor_id,
        "bucket": SETTINGS.influxdb_bucket,
        "org": SETTINGS.influxdb_org,
        "measurement": SETTINGS.influxdb_measurement,
        "timezone": SETTINGS.timezone_name,
        "latest": latest,
        "condition": classify_status(effective_voltage(latest)) if latest else "No data",
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
    points = await asyncio.to_thread(repository.fetch_history, parsed_sensor_id, target_day, interval_minutes)
    latest = points[-1] if points else None

    return {
        "sensor_id": parsed_sensor_id,
        "day": target_day.isoformat(),
        "interval_minutes": interval_minutes,
        "timezone": SETTINGS.timezone_name,
        "points": points,
        "latest": latest,
        "condition": classify_status(effective_voltage(latest)) if latest else "No data",
    }


@app.get("/api/insights")
def api_insights() -> dict[str, Any]:
    insights = read_latest_insights()
    if insights is None:
        return {
            "available": False,
            "timezone": SETTINGS.timezone_name,
            "message": "ML insights are not ready yet",
        }
    insights["available"] = True
    return insights


@app.get("/api/analytics")
async def api_analytics(request: Request, sensor_id: str | None = None) -> dict[str, Any]:
    repository = get_repository(request)
    parsed_sensor_id = parse_sensor_id(sensor_id)
    recent_points = await asyncio.to_thread(repository.fetch_recent, parsed_sensor_id, timedelta(hours=1))
    latest_insights = read_latest_insights()
    analytics = build_analytics_payload(recent_points, latest_insights)
    analytics["sensor_id"] = parsed_sensor_id
    return analytics


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
                yield format_sse("telemetry", latest)
            else:
                yield format_sse(
                    "heartbeat",
                    {"at": utc_now().isoformat().replace("+00:00", "Z")},
                )

            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
