"""
FetalGuard AI — FastAPI Prediction Service
Phase 08: Deployment & MLOps

Endpoints:
  GET  /health          — liveness probe
  GET  /info            — model metadata
  POST /predict         — single prediction (top-5 features)
  POST /predict/full    — full 22-feature prediction
  POST /predict/batch   — batch predictions
  GET  /metrics         — running prediction statistics
"""

import time
import pickle
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
log = logging.getLogger("ctg_api")

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent          # ctg_mlops/
CONFIG_PATH   = PROJECT_ROOT / "config" / "config.yaml"
ARTIFACT_PATH = PROJECT_ROOT / "deployment_artifact.pkl"

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# ── Global state (loaded once at startup via lifespan) ────────────────────────
_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact and config once at startup; release at shutdown."""
    log.info("Loading model artifact …")
    cfg = load_config()

    # Always use the module-level ARTIFACT_PATH (absolute, resolved at import time).
    # This allows tests to patch app_module.ARTIFACT_PATH before importing the app.
    artifact_path = ARTIFACT_PATH

    if not artifact_path.exists():
        raise RuntimeError(
            f"Artifact not found at {artifact_path}. "
            "Run build_artifact.py first."
        )

    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)

    _state["model"]     = artifact["model"]
    _state["scaler"]    = artifact["scaler"]
    _state["features"]  = artifact["feature_names"]
    _state["top5"]      = artifact["top5_features"]
    _state["stats"]     = artifact["feature_stats"]
    _state["threshold"] = cfg["model"]["optimal_threshold"]
    _state["version"]   = cfg["model"]["version"]
    _state["name"]      = cfg["model"]["name"]

    # Running counters for /metrics endpoint
    _state["request_count"]    = 0
    _state["high_risk_count"]  = 0
    _state["normal_count"]     = 0
    _state["total_latency_ms"] = 0.0
    _state["startup_time"]     = time.time()

    log.info(
        f"Model loaded: {_state['name']} v{_state['version']}  "
        f"| features={len(_state['features'])}  "
        f"| threshold={_state['threshold']}"
    )
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FetalGuard AI API",
    description=(
        "Clinical decision-support API for cardiotocography fetal risk classification. "
        "XGBoost model trained on the UCI CTG dataset. "
        "For research/demonstration purposes only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Top-5 SHAP feature input. Missing features default to training-set medians.
    All values must be non-negative; ASTV and ALTV must be percentages (0–100).
    """
    ASTV     : Optional[float] = Field(None, ge=0, le=100,  description="% time with abnormal STV")
    AC       : Optional[float] = Field(None, ge=0,          description="Accelerations count")
    Variance : Optional[float] = Field(None, ge=0,          description="FHR histogram variance")
    ALTV     : Optional[float] = Field(None, ge=0, le=100,  description="% time with abnormal LTV")
    Width    : Optional[float] = Field(None, ge=0,          description="FHR histogram width")
    threshold: Optional[float] = Field(None, ge=0.01, le=0.99,
                                       description="Override decision threshold (default: config value)")

    @field_validator("AC", "Variance", "Width", mode="before")
    @classmethod
    def must_be_non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError("Feature value must be ≥ 0")
        return v


class FullPredictRequest(BaseModel):
    """Full 22-feature prediction request. All features optional; missing → median."""
    LB       : Optional[float] = Field(None, ge=50,  le=250)
    AC       : Optional[float] = Field(None, ge=0)
    FM       : Optional[float] = Field(None, ge=0)
    UC       : Optional[float] = Field(None, ge=0)
    ASTV     : Optional[float] = Field(None, ge=0,  le=100)
    MSTV     : Optional[float] = Field(None, ge=0)
    ALTV     : Optional[float] = Field(None, ge=0,  le=100)
    MLTV     : Optional[float] = Field(None, ge=0)
    DL       : Optional[float] = Field(None, ge=0)
    DS       : Optional[float] = Field(None, ge=0)
    DP       : Optional[float] = Field(None, ge=0)
    DR       : Optional[float] = Field(None, ge=0)
    Width    : Optional[float] = Field(None, ge=0)
    Min      : Optional[float] = Field(None, ge=0)
    Max      : Optional[float] = Field(None, ge=0)
    Nmax     : Optional[float] = Field(None, ge=0)
    Nzeros   : Optional[float] = Field(None, ge=0)
    Mode     : Optional[float] = Field(None, ge=0)
    Mean     : Optional[float] = Field(None, ge=0)
    Median   : Optional[float] = Field(None, ge=0)
    Variance : Optional[float] = Field(None, ge=0)
    Tendency : Optional[float] = Field(None)
    threshold: Optional[float] = Field(None, ge=0.01, le=0.99)


class PredictionResponse(BaseModel):
    prediction  : str
    probability : float
    threshold   : float
    model_version: str


class BatchPredictRequest(BaseModel):
    inputs: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    results      : list[PredictionResponse]
    count        : int
    high_risk_n  : int
    normal_n     : int


# ── Core prediction helper ────────────────────────────────────────────────────
def _run_prediction(inputs: dict, threshold: float) -> PredictionResponse:
    features = _state["features"]
    stats    = _state["stats"]
    scaler   = _state["scaler"]
    model    = _state["model"]

    row = {f: inputs.get(f, stats[f]["median"]) for f in features}
    df  = pd.DataFrame([row], columns=features)

    X_scaled = scaler.transform(df)
    proba    = float(model.predict_proba(X_scaled)[0, 1])
    label    = "High Risk" if proba >= threshold else "Normal"

    return PredictionResponse(
        prediction   = label,
        probability  = round(proba, 6),
        threshold    = threshold,
        model_version= _state["version"],
    )


# ── Middleware: request logging & latency tracking ────────────────────────────
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000
    _state["total_latency_ms"] = _state.get("total_latency_ms", 0) + latency_ms
    log.info(
        f"{request.method} {request.url.path}  "
        f"status={response.status_code}  "
        f"latency={latency_ms:.1f}ms"
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Ops"])
def health():
    """Liveness probe — returns 200 if the service is running."""
    return {
        "status"   : "ok",
        "model"    : _state.get("name", "unknown"),
        "version"  : _state.get("version", "unknown"),
        "uptime_s" : round(time.time() - _state.get("startup_time", time.time()), 1),
    }


@app.get("/info", tags=["Ops"])
def info():
    """Model metadata — feature list, threshold, version."""
    return {
        "model_name"   : _state["name"],
        "model_version": _state["version"],
        "features"     : _state["features"],
        "top5_features": _state["top5"],
        "threshold"    : _state["threshold"],
        "description"  : (
            "XGBoost classifier trained on UCI Cardiotocography dataset. "
            "Threshold selected to minimise clinical cost (FP×1 + FN×10)."
        ),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Single prediction using top-5 SHAP features.
    Missing features default to training-set medians.
    """
    t = req.threshold if req.threshold is not None else _state["threshold"]
    inputs = req.model_dump(exclude={"threshold"}, exclude_none=False)
    inputs = {k: v for k, v in inputs.items() if v is not None}

    result = _run_prediction(inputs, t)

    # Update running counters
    _state["request_count"] += 1
    if result.prediction == "High Risk":
        _state["high_risk_count"] += 1
    else:
        _state["normal_count"] += 1

    log.info(
        f"predict | ASTV={inputs.get('ASTV', '?')}  "
        f"AC={inputs.get('AC', '?')}  "
        f"→ {result.prediction} (p={result.probability:.4f})"
    )
    return result


@app.post("/predict/full", response_model=PredictionResponse, tags=["Prediction"])
def predict_full(req: FullPredictRequest):
    """
    Single prediction with all 22 CTG features.
    Missing features default to training-set medians.
    """
    t = req.threshold if req.threshold is not None else _state["threshold"]
    inputs = {
        k: v for k, v in req.model_dump(exclude={"threshold"}).items()
        if v is not None
    }
    result = _run_prediction(inputs, t)
    _state["request_count"] += 1
    if result.prediction == "High Risk":
        _state["high_risk_count"] += 1
    else:
        _state["normal_count"] += 1
    return result


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(req: BatchPredictRequest):
    """
    Batch predictions — up to 500 patients per request.
    Each row uses the same top-5 feature schema as /predict.
    """
    t = _state["threshold"]
    results = []
    for item in req.inputs:
        item_t  = item.threshold if item.threshold is not None else t
        inputs  = {k: v for k, v in item.model_dump(exclude={"threshold"}).items()
                   if v is not None}
        results.append(_run_prediction(inputs, item_t))

    high_risk_n = sum(1 for r in results if r.prediction == "High Risk")
    normal_n    = len(results) - high_risk_n
    _state["request_count"] += len(results)
    _state["high_risk_count"] += high_risk_n
    _state["normal_count"]    += normal_n

    return BatchPredictionResponse(
        results     = results,
        count       = len(results),
        high_risk_n = high_risk_n,
        normal_n    = normal_n,
    )


@app.get("/metrics", tags=["Ops"])
def metrics():
    """Running prediction statistics since last restart — for basic monitoring."""
    total   = _state["request_count"]
    avg_lat = (
        _state["total_latency_ms"] / max(total, 1)
    )
    return {
        "total_predictions"  : total,
        "high_risk_count"    : _state["high_risk_count"],
        "normal_count"       : _state["normal_count"],
        "high_risk_rate"     : round(_state["high_risk_count"] / max(total, 1), 4),
        "avg_latency_ms"     : round(avg_lat, 2),
        "uptime_s"           : round(time.time() - _state.get("startup_time", time.time()), 1),
    }
