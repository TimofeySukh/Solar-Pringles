from __future__ import annotations

import asyncio
import json
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

    def _latest_query(self, sensor_id: str) -> str:
        return f"""
from(bucket: "{self.settings.influxdb_bucket}")
  |> range(start: -12h)
  |> filter(fn: (r) => r["_measurement"] == "{self.settings.influxdb_measurement}")
  |> filter(fn: (r) => r["sensor_id"] == "{sensor_id}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "sensor_id", "raw_voltage", "smoothed_voltage"])
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 1)
"""

    def _history_query(self, sensor_id: str, start: datetime, stop: datetime, every_minutes: int) -> str:
        every = f"{every_minutes}m"
        start_iso = start.isoformat().replace("+00:00", "Z")
        stop_iso = stop.isoformat().replace("+00:00", "Z")
        return f"""
from(bucket: "{self.settings.influxdb_bucket}")
  |> range(start: time(v: "{start_iso}"), stop: time(v: "{stop_iso}"))
  |> filter(fn: (r) => r["_measurement"] == "{self.settings.influxdb_measurement}")
  |> filter(fn: (r) => r["sensor_id"] == "{sensor_id}")
  |> aggregateWindow(every: {every}, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "sensor_id", "raw_voltage", "smoothed_voltage"])
  |> sort(columns: ["_time"])
"""

    @staticmethod
    def _row_to_point(record: Any) -> dict[str, Any]:
        raw_voltage = record.values.get("raw_voltage")
        smoothed_voltage = record.values.get("smoothed_voltage")
        recorded_at = record.get_time()
        recorded_at_utc = recorded_at.astimezone(UTC)
        recorded_at_local = recorded_at_utc.astimezone(LOCAL_TIMEZONE)

        return {
            "timestamp": recorded_at_utc.isoformat().replace("+00:00", "Z"),
            "timestamp_local": recorded_at_local.isoformat(),
            "sensor_id": record.values.get("sensor_id"),
            "raw_voltage": float(raw_voltage) if raw_voltage is not None else None,
            "smoothed_voltage": float(smoothed_voltage) if smoothed_voltage is not None else None,
        }

    def fetch_latest(self, sensor_id: str) -> dict[str, Any] | None:
        tables = self.query_api.query(self._latest_query(sensor_id))
        for table in tables:
            for record in table.records:
                return self._row_to_point(record)
        return None

    def fetch_history(self, sensor_id: str, target_day: date, every_minutes: int) -> list[dict[str, Any]]:
        start_local = datetime.combine(target_day, time.min, tzinfo=LOCAL_TIMEZONE)
        stop_local = start_local + timedelta(days=1)
        start = start_local.astimezone(UTC)
        stop = stop_local.astimezone(UTC)
        tables = self.query_api.query(self._history_query(sensor_id, start, stop, every_minutes))
        rows: list[dict[str, Any]] = []
        for table in tables:
            rows.extend(self._row_to_point(record) for record in table.records)
        return rows


app = FastAPI(title=SETTINGS.api_title, version="0.2.0")
app.state.settings = SETTINGS
app.state.influx = None


@app.on_event("startup")
def startup_event() -> None:
    app.state.influx = InfluxRepository(SETTINGS)


@app.on_event("shutdown")
def shutdown_event() -> None:
    influx = app.state.influx
    if influx is not None:
        influx.close()


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


def effective_voltage(point: dict[str, Any]) -> float | None:
    smoothed = point.get("smoothed_voltage")
    if smoothed is not None:
        return smoothed
    return point.get("raw_voltage")


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


def local_now() -> datetime:
    return datetime.now(LOCAL_TIMEZONE)


def read_latest_insights() -> dict[str, Any] | None:
    insights_path = Path(SETTINGS.model_registry_dir) / "insights" / "latest.json"
    if not insights_path.exists():
        return None
    return json.loads(insights_path.read_text(encoding="utf-8"))


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
