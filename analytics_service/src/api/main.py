from __future__ import annotations

from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

openapi_tags = [
    {"name": "health", "description": "Service health and diagnostics."},
    {"name": "internal", "description": "Internal APIs used by backend_api_service."},
]


app = FastAPI(
    title="Energy Insights Platform - Analytics Service",
    description=(
        "Internal analytics microservice.\n\n"
        "Provides time-series aggregation, baseline calculation, and anomaly scoring.\n"
        "This service is intended to be called by backend_api_service."
    ),
    version="0.3.0",
    openapi_tags=openapi_tags,
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
    granularity: str = Field(default="daily", description="daily|weekly|monthly")
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


def _calc_anomaly_score(values: List[float]) -> float:
    """Compute a simple z-score-like anomaly score using last point deviation."""
    if len(values) < 5:
        return 0.0
    mu = mean(values[:-1])
    sigma = pstdev(values[:-1]) or 0.0
    if sigma == 0.0:
        return 0.0
    return abs((values[-1] - mu) / sigma)


@app.post(
    "/internal/analyze",
    tags=["internal"],
    summary="Analyze readings window",
    operation_id="internal_analyze",
    response_model=AnalyzeResponse,
)
# PUBLIC_INTERFACE
def internal_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze time-series readings for baseline and anomaly scoring.

    This implementation is intentionally lightweight:
    - Baseline: mean of values excluding last point
    - Anomaly score: deviation of last point from baseline in stddev units

    Args:
        req: AnalyzeRequest

    Returns:
        AnalyzeResponse
    """
    # Normalize input
    points: List[ReadingPoint] = [ReadingPoint(**r) for r in req.readings]
    values = [float(p.value) for p in points]

    baseline = mean(values[:-1]) if len(values) > 1 else values[0]
    score = _calc_anomaly_score(values)

    output = {
        "baseline": baseline,
        "last_value": values[-1],
        "n_points": len(values),
        "window_start": req.window_start.isoformat(),
        "window_end": req.window_end.isoformat(),
    }

    should_alert = score >= 3.0
    severity = "critical" if score >= 5.0 else "warning" if score >= 3.0 else "info"

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
