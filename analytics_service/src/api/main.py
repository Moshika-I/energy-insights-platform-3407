from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.core.db import get_db, lifespan

openapi_tags = [
    {"name": "health", "description": "Service health and diagnostics."},
    {"name": "internal", "description": "Internal APIs used by backend_api_service."},
    {"name": "analytics", "description": "Internal analytics primitives (anomalies, benchmarks)."},
]

Granularity = Literal["daily", "weekly", "monthly"]


app = FastAPI(
    title="Energy Insights Platform - Analytics Service",
    description=(
        "Internal analytics microservice.\n\n"
        "Provides time-series aggregation, baseline calculation, anomaly scoring, and benchmarking.\n"
        "This service is intended to be called by backend_api_service."
    ),
    version="0.3.0",
    openapi_tags=openapi_tags,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"], summary="Health check", operation_id="health_check")
# PUBLIC_INTERFACE
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"message": "Healthy"}


class ReadingPoint(BaseModel):
    """A simplified reading point passed from backend_api_service."""

    reading_at: datetime = Field(..., description="Timestamp.")
    value: float = Field(..., description="Consumption value.")


class AnalyzeRequest(BaseModel):
    """Analyze a window of readings and return anomaly/baseline outputs."""

    tenant_id: str = Field(..., description="Tenant UUID.")
    meter_id: str = Field(..., description="Meter UUID.")
    window_start: datetime = Field(..., description="Start time.")
    window_end: datetime = Field(..., description="End time.")
    granularity: Granularity = Field(default="daily", description="daily|weekly|monthly")
    readings: List[Dict[str, Any]] = Field(..., description="List of readings (each contains reading_at, value).")


class AnalyzeResponse(BaseModel):
    """Analysis result for persistence by backend_api_service."""

    output_type: str = Field(..., description="Type of analytic output.")
    model_version: str = Field(..., description="Algorithm/model version.")
    score: Optional[float] = Field(default=None, description="Anomaly score.")
    output: Dict[str, Any] = Field(..., description="Payload (JSON).")
    should_alert: bool = Field(default=False, description="Whether backend should trigger an alert.")
    alert_severity: Optional[str] = Field(default=None, description="info|warning|critical")
    alert_title: Optional[str] = Field(default=None, description="Suggested alert title")
    alert_message: Optional[str] = Field(default=None, description="Suggested alert message")


class AnomalyQuery(BaseModel):
    """DB-backed anomaly query parameters."""

    tenant_id: str = Field(..., description="Tenant UUID.")
    meter_id: str = Field(..., description="Meter UUID.")
    window_start: datetime = Field(..., description="Start time.")
    window_end: datetime = Field(..., description="End time.")
    granularity: Granularity = Field(default="daily", description="daily|weekly|monthly")


class AnomalyOut(BaseModel):
    """Anomaly output derived from readings window."""

    meter_id: str = Field(..., description="Meter UUID.")
    window_start: datetime
    window_end: datetime
    baseline: float
    last_value: float
    score: float
    should_alert: bool
    severity: str


class BenchmarkQuery(BaseModel):
    """Benchmark query parameters.

    Benchmark compares a meter's average daily usage over a window to its tenant peers.
    """

    tenant_id: str = Field(..., description="Tenant UUID.")
    meter_id: str = Field(..., description="Meter UUID.")
    window_start: datetime = Field(..., description="Start time.")
    window_end: datetime = Field(..., description="End time.")


class BenchmarkOut(BaseModel):
    """Benchmark response."""

    tenant_id: str
    meter_id: str
    window_start: datetime
    window_end: datetime
    your_avg_daily: float
    peer_median_daily: float
    percentile: float
    peer_count: int


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware UTC for consistent bucketing."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _calc_anomaly_score(values: List[float]) -> float:
    """Compute a simple z-score-like anomaly score using last point deviation."""
    if len(values) < 5:
        return 0.0
    mu = mean(values[:-1])
    sigma = pstdev(values[:-1]) or 0.0
    if sigma == 0.0:
        return 0.0
    return abs((values[-1] - mu) / sigma)


def _severity_for_score(score: float) -> str:
    """Map score to severity."""
    if score >= 5.0:
        return "critical"
    if score >= 3.0:
        return "warning"
    return "info"


def _median(values: List[float]) -> float:
    """Compute median without external deps."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _percentile_rank(values: List[float], x: float) -> float:
    """Return percentile rank of x within values in [0,100]."""
    if not values:
        return 0.0
    # "weak" definition: count <= x
    le = sum(1 for v in values if v <= x)
    return float(le) / float(len(values)) * 100.0


async def _fetch_readings_window(tenant_id: str, meter_id: str, window_start: datetime, window_end: datetime) -> List[Tuple[datetime, float]]:
    """Fetch readings for a window."""
    db = get_db()
    rows = await db.fetch_all(
        """
        SELECT reading_at, value
        FROM meter_readings
        WHERE tenant_id = :tenant_id::uuid
          AND meter_id = :meter_id::uuid
          AND reading_at >= :window_start
          AND reading_at <= :window_end
        ORDER BY reading_at ASC
        """,
        {
            "tenant_id": tenant_id,
            "meter_id": meter_id,
            "window_start": _ensure_utc(window_start).isoformat(),
            "window_end": _ensure_utc(window_end).isoformat(),
        },
    )
    out: List[Tuple[datetime, float]] = []
    for r in rows:
        ra = r.get("reading_at")
        val = r.get("value")
        try:
            reading_at = ra if isinstance(ra, datetime) else datetime.fromisoformat(str(ra).replace("Z", "+00:00"))
        except Exception:
            continue
        out.append((reading_at, float(val)))
    return out


@app.post(
    "/internal/analyze",
    tags=["internal"],
    summary="Analyze readings window (called by backend_api_service)",
    operation_id="internal_analyze",
    response_model=AnalyzeResponse,
)
# PUBLIC_INTERFACE
def internal_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze time-series readings for baseline and anomaly scoring.

    This endpoint is used by backend_api_service orchestration.

    Args:
        req: AnalyzeRequest

    Returns:
        AnalyzeResponse
    """
    points: List[ReadingPoint] = [ReadingPoint(**r) for r in req.readings]
    values = [float(p.value) for p in points]

    baseline = mean(values[:-1]) if len(values) > 1 else values[0]
    score = _calc_anomaly_score(values)
    severity = _severity_for_score(score)

    output = {
        "baseline": baseline,
        "last_value": values[-1],
        "n_points": len(values),
        "window_start": _ensure_utc(req.window_start).isoformat(),
        "window_end": _ensure_utc(req.window_end).isoformat(),
        "granularity": req.granularity,
    }

    should_alert = score >= 3.0

    return AnalyzeResponse(
        output_type="baseline_anomaly",
        model_version="simple_v1",
        score=score,
        output=output,
        should_alert=should_alert,
        alert_severity=severity if should_alert else None,
        alert_title="Consumption anomaly detected" if should_alert else None,
        alert_message=(
            f"Anomaly score {score:.2f}. Last value {values[-1]:.2f} vs baseline {baseline:.2f}."
            if should_alert
            else None
        ),
    )


@app.get(
    "/internal/anomalies",
    tags=["analytics"],
    summary="Compute anomaly score for a meter using DB readings",
    operation_id="internal_get_anomalies",
    response_model=AnomalyOut,
)
# PUBLIC_INTERFACE
async def internal_get_anomalies(
    tenant_id: str = Query(..., description="Tenant UUID."),
    meter_id: str = Query(..., description="Meter UUID."),
    window_start: datetime = Query(..., description="Start time."),
    window_end: datetime = Query(..., description="End time."),
    granularity: Granularity = Query(default="daily", description="daily|weekly|monthly"),
) -> AnomalyOut:
    """Compute anomaly score for a meter using readings in DB."""
    readings = await _fetch_readings_window(tenant_id, meter_id, window_start, window_end)
    if len(readings) < 2:
        raise HTTPException(status_code=404, detail="Not enough readings for anomaly computation")

    values = [v for _, v in readings]
    baseline = mean(values[:-1]) if len(values) > 1 else values[0]
    score = _calc_anomaly_score(values)
    severity = _severity_for_score(score)
    should_alert = score >= 3.0

    # granularity is kept for API compatibility; computation is window-based
    _ = granularity

    return AnomalyOut(
        meter_id=meter_id,
        window_start=_ensure_utc(window_start),
        window_end=_ensure_utc(window_end),
        baseline=float(baseline),
        last_value=float(values[-1]),
        score=float(score),
        should_alert=bool(should_alert),
        severity=severity,
    )


@app.get(
    "/internal/benchmark",
    tags=["analytics"],
    summary="Benchmark a meter against tenant peers",
    operation_id="internal_get_benchmark",
    response_model=BenchmarkOut,
)
# PUBLIC_INTERFACE
async def internal_get_benchmark(
    tenant_id: str = Query(..., description="Tenant UUID."),
    meter_id: str = Query(..., description="Meter UUID."),
    window_start: datetime = Query(..., description="Start time."),
    window_end: datetime = Query(..., description="End time."),
) -> BenchmarkOut:
    """Benchmark a meter vs peers within tenant over the provided window.

    Implementation:
    - For each meter in tenant, compute avg daily usage over window
      (total usage / number_of_days_in_window).
    - Percentile is computed against peer distribution (including your meter).
    """
    ws = _ensure_utc(window_start)
    we = _ensure_utc(window_end)
    if we < ws:
        raise HTTPException(status_code=422, detail="window_end must be >= window_start")

    # Avoid division-by-zero; treat as at least 1 day.
    n_days = max(1.0, (we - ws).total_seconds() / float(timedelta(days=1).total_seconds()))

    db = get_db()
    rows = await db.fetch_all(
        """
        WITH totals AS (
          SELECT mr.meter_id::text AS meter_id,
                 SUM(mr.value)::float AS total_value
          FROM meter_readings mr
          JOIN meters m ON m.id = mr.meter_id
          WHERE mr.tenant_id = :tenant_id::uuid
            AND mr.reading_at >= :window_start
            AND mr.reading_at <= :window_end
          GROUP BY mr.meter_id
        )
        SELECT meter_id, total_value
        FROM totals
        """,
        {"tenant_id": tenant_id, "window_start": ws.isoformat(), "window_end": we.isoformat()},
    )

    if not rows:
        raise HTTPException(status_code=404, detail="No readings found for tenant window")

    per_meter_daily: Dict[str, float] = {}
    for r in rows:
        mid = str(r.get("meter_id"))
        total = float(r.get("total_value") or 0.0)
        per_meter_daily[mid] = total / n_days

    if meter_id not in per_meter_daily:
        raise HTTPException(status_code=404, detail="No readings found for requested meter in window")

    dist = list(per_meter_daily.values())
    your = per_meter_daily[meter_id]
    peer_median = _median(dist)
    percentile = _percentile_rank(dist, your)

    return BenchmarkOut(
        tenant_id=tenant_id,
        meter_id=meter_id,
        window_start=ws,
        window_end=we,
        your_avg_daily=float(your),
        peer_median_daily=float(peer_median),
        percentile=float(percentile),
        peer_count=len(dist),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Any, exc: Exception) -> JSONResponse:
    """Return predictable error response for clients."""
    return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(exc)})
