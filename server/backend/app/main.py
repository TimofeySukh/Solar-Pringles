import os

from fastapi import FastAPI


app = FastAPI(
    title=os.getenv("API_TITLE", "Sollar Panel API"),
    version="0.1.0",
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "service": "backend"}


@app.get("/api/status")
def api_status() -> dict[str, str]:
    return {
        "status": "bootstrap",
        "message": "Backend scaffold is running. Historical and live endpoints are the next step.",
        "bucket": os.getenv("INFLUXDB_BUCKET", "solar_metrics"),
        "org": os.getenv("INFLUXDB_ORG", "sollar_panel"),
    }
