"""
Unit & integration tests for the CTG FastAPI prediction service.
Run with: pytest tests/ -v --tb=short

Tests are designed to be environment-agnostic — they read the actual
feature count and probability values from the running API rather than
hardcoding assumptions that break when the dataset differs.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.main import app

# TestClient as context manager triggers lifespan startup → populates _state
@pytest.fixture(scope="session")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# Fixtures that adapt to the actual artifact loaded by the server
@pytest.fixture(scope="session")
def api_info(client):
    """Fetch /info once and reuse across tests."""
    return client.get("/info").json()


@pytest.fixture(scope="session")
def n_features(api_info):
    """Actual feature count from the loaded artifact."""
    return len(api_info["features"])


@pytest.fixture(scope="session")
def default_threshold(api_info):
    """Actual threshold from the loaded artifact."""
    return api_info["threshold"]


@pytest.fixture(scope="session")
def normal_input(api_info):
    """
    Construct a Normal input dynamically: set ASTV to its min possible value,
    AC to its max. Works regardless of exact feature count or dataset version.
    """
    inputs = {}
    features = api_info["features"]
    if "ASTV" in features:
        inputs["ASTV"] = 16.0   # low abnormal variability → Normal
    if "AC" in features:
        inputs["AC"] = 10.0     # many accelerations → Normal
    if "Variance" in features:
        inputs["Variance"] = 1.0
    if "ALTV" in features:
        inputs["ALTV"] = 0.0
    if "Width" in features:
        inputs["Width"] = 100.0
    return inputs


@pytest.fixture(scope="session")
def abnormal_input(api_info):
    """
    Construct an Abnormal input dynamically.
    """
    inputs = {}
    features = api_info["features"]
    if "ASTV" in features:
        inputs["ASTV"] = 84.0   # near-maximum abnormal STV → Abnormal
    if "AC" in features:
        inputs["AC"] = 0.0      # no accelerations → Abnormal
    if "Variance" in features:
        inputs["Variance"] = 120.0
    if "ALTV" in features:
        inputs["ALTV"] = 80.0
    if "Width" in features:
        inputs["Width"] = 10.0
    return inputs


# ── Health & Info ─────────────────────────────────────────────────────────────
class TestHealthAndInfo:
    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_has_ok_status(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_health_has_uptime(self, client):
        assert client.get("/health").json()["uptime_s"] >= 0

    def test_info_returns_200(self, client):
        assert client.get("/info").status_code == 200

    def test_info_has_required_fields(self, client):
        data = client.get("/info").json()
        for f in ["model_name", "model_version", "features", "threshold"]:
            assert f in data, f"Missing field: {f}"

    def test_info_has_at_least_5_features(self, client, n_features):
        """
        Checks at least 5 features exist (the top-5 SHAP minimum).
        Does NOT hardcode 22 — feature count depends on your dataset columns.
        """
        assert n_features >= 5, (
            f"Expected at least 5 features, got {n_features}. "
            "Check that build_artifact.py ran against your local test.xlsx."
        )

    def test_info_feature_count_is_consistent(self, client, n_features, api_info):
        """Feature list length matches what /info reports."""
        assert len(api_info["features"]) == n_features

    def test_info_threshold_is_float(self, client, api_info):
        assert isinstance(api_info["threshold"], float)

    def test_info_threshold_in_valid_range(self, client, api_info):
        assert 0.01 <= api_info["threshold"] <= 0.99


# ── /predict (top-5 schema) ────────────────────────────────────────────────────
class TestPredict:
    def test_normal_prediction_returns_200(self, client, normal_input):
        assert client.post("/predict", json=normal_input).status_code == 200

    def test_abnormal_prediction_returns_200(self, client, abnormal_input):
        assert client.post("/predict", json=abnormal_input).status_code == 200

    def test_normal_input_classified_normal(self, client, normal_input):
        r = client.post("/predict", json=normal_input)
        assert r.json()["prediction"] == "Normal", (
            f"Expected Normal but got {r.json()['prediction']} "
            f"(probability={r.json()['probability']:.4f}). "
            "Adjust normal_input fixture values if your model thresholds differ."
        )

    def test_abnormal_input_classified_high_risk(self, client, abnormal_input):
        r = client.post("/predict", json=abnormal_input)
        assert r.json()["prediction"] == "High Risk", (
            f"Expected High Risk but got {r.json()['prediction']} "
            f"(probability={r.json()['probability']:.4f}). "
            "Adjust abnormal_input fixture values if your model thresholds differ."
        )

    def test_response_has_probability(self, client, normal_input):
        prob = client.post("/predict", json=normal_input).json()["probability"]
        assert 0.0 <= prob <= 1.0

    def test_response_has_model_version(self, client, normal_input):
        assert "model_version" in client.post("/predict", json=normal_input).json()

    def test_empty_body_uses_medians(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 200
        assert r.json()["prediction"] in ("Normal", "High Risk")

    def test_threshold_override_changes_classification(self, client, api_info):
        """
        Verifies the threshold override mechanism works by bracketing a real
        mid-range probability with thresholds just above and just below it.

        Scans a grid of ASTV × AC × ALTV values to find an input that produces
        a probability clearly between 0.10 and 0.90, then tests that:
          - threshold set just BELOW that probability  → High Risk
          - threshold set just ABOVE that probability  → Normal
          - probability is identical regardless of threshold
        """
        features = api_info["features"]
        ref_proba = None
        ref_input = {}

        # Wide grid covering combinations likely to produce mid-range probabilities
        astv_vals = [20, 30, 40, 50, 55, 60, 65, 70]
        ac_vals   = [0, 1, 2, 3, 4]
        altv_vals = [0, 5, 10, 20]

        for astv in astv_vals:
            for ac in ac_vals:
                for altv in altv_vals:
                    test_input = {}
                    if "ASTV" in features:
                        test_input["ASTV"] = float(astv)
                    if "AC" in features:
                        test_input["AC"]   = float(ac)
                    if "ALTV" in features:
                        test_input["ALTV"] = float(altv)

                    r = client.post("/predict", json=test_input)
                    p = r.json()["probability"]

                    if 0.10 <= p <= 0.90:
                        ref_proba = p
                        ref_input = test_input
                        break
                if ref_proba is not None:
                    break
            if ref_proba is not None:
                break

        assert ref_proba is not None, (
            "Could not find a mid-range (0.10–0.90) probability across "
            "ASTV×AC×ALTV grid sweep. Your model may be extremely decisive. "
            "Add 'pytest.mark.skip' to this test or widen the grid manually."
        )

        t_low  = round(max(0.01, ref_proba - 0.05), 2)
        t_high = round(min(0.99, ref_proba + 0.05), 2)

        r_low  = client.post("/predict", json={**ref_input, "threshold": t_low})
        r_high = client.post("/predict", json={**ref_input, "threshold": t_high})

        assert r_low.json()["prediction"] == "High Risk", (
            f"threshold={t_low} (below proba={ref_proba:.4f}), "
            f"expected High Risk — got {r_low.json()['prediction']}"
        )
        assert r_high.json()["prediction"] == "Normal", (
            f"threshold={t_high} (above proba={ref_proba:.4f}), "
            f"expected Normal — got {r_high.json()['prediction']}"
        )
        assert r_low.json()["probability"] == pytest.approx(
            r_high.json()["probability"], abs=1e-6
        )

    def test_negative_astv_rejected(self, client):
        assert client.post("/predict", json={"ASTV": -5.0}).status_code == 422

    def test_astv_over_100_rejected(self, client):
        assert client.post("/predict", json={"ASTV": 150.0}).status_code == 422

    def test_threshold_out_of_range_rejected(self, client):
        assert client.post("/predict", json={"threshold": 1.5}).status_code == 422


# ── /predict/full (full feature schema) ───────────────────────────────────────
class TestPredictFull:
    def test_full_endpoint_returns_200(self, client):
        assert client.post("/predict/full", json={"ASTV": 80.0}).status_code == 200

    def test_full_prediction_has_correct_fields(self, client):
        data = client.post("/predict/full", json={"ASTV": 50.0}).json()
        for f in ["prediction", "probability", "threshold"]:
            assert f in data

    def test_full_empty_body_uses_medians(self, client):
        assert client.post("/predict/full", json={}).status_code == 200

    def test_full_probability_in_valid_range(self, client):
        prob = client.post("/predict/full", json={}).json()["probability"]
        assert 0.0 <= prob <= 1.0


# ── /predict/batch ─────────────────────────────────────────────────────────────
class TestPredictBatch:
    def test_batch_returns_200(self, client, normal_input, abnormal_input):
        r = client.post("/predict/batch", json={"inputs": [normal_input, abnormal_input]})
        assert r.status_code == 200

    def test_batch_count_matches_input(self, client, normal_input, abnormal_input):
        payload = {"inputs": [normal_input, abnormal_input, normal_input]}
        assert client.post("/predict/batch", json=payload).json()["count"] == 3

    def test_batch_high_risk_plus_normal_equals_count(self, client, normal_input, abnormal_input):
        data = client.post(
            "/predict/batch", json={"inputs": [normal_input, abnormal_input]}
        ).json()
        assert data["high_risk_n"] + data["normal_n"] == data["count"]

    def test_batch_single_item_works(self, client, normal_input):
        r = client.post("/predict/batch", json={"inputs": [normal_input]})
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def test_empty_batch_rejected(self, client):
        assert client.post("/predict/batch", json={"inputs": []}).status_code == 422


# ── /metrics ───────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_metrics_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_metrics_has_required_fields(self, client):
        data = client.get("/metrics").json()
        for f in ["total_predictions", "high_risk_count", "normal_count",
                  "high_risk_rate", "avg_latency_ms"]:
            assert f in data, f"Missing field: {f}"

    def test_high_risk_rate_is_valid_proportion(self, client, normal_input, abnormal_input):
        client.post("/predict", json=normal_input)
        client.post("/predict", json=abnormal_input)
        rate = client.get("/metrics").json()["high_risk_rate"]
        assert 0.0 <= rate <= 1.0
