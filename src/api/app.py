"""
Flask Backend — Earthquake ML API
===================================
Endpoints:
  POST /predict          → magnitude prediction
  GET  /analytics/region → per-region stats
  GET  /analytics/time   → monthly trends
  GET  /analytics/anomaly→ z-score anomalies
  GET  /data             → full processed dataset
  GET  /model/metrics    → model eval metrics
  GET  /regions          → list of all regions
  GET  /health           → health check
"""

import os
import sys
import json
import logging
import traceback

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── Setup ────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# BASE_DIR is 'src/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from core.regions import REGION_NAMES, classify_region, get_region_center

MODEL_PATH = os.path.join(ROOT_DIR, "models", "model.pkl")
FEATURES_PATH = os.path.join(ROOT_DIR, "models", "feature_columns.json")
ENCODER_PATH = os.path.join(ROOT_DIR, "models", "region_encoder.pkl")
METRICS_PATH = os.path.join(ROOT_DIR, "models", "metrics.json")
TRAIN_PATH = os.path.join(ROOT_DIR, "data", "processed", "train.csv")
TEST_PATH = os.path.join(ROOT_DIR, "data", "processed", "test.csv")

# ─── Resource Cache ───────────────────────────────────────────
_cache = {}


def _load():
    """Load model, encoder, features, and data into memory once."""
    if _cache.get("loaded"):
        return
    try:
        _cache["model"] = joblib.load(MODEL_PATH)
        _cache["encoder"] = joblib.load(ENCODER_PATH)
        with open(FEATURES_PATH) as f:
            _cache["features"] = json.load(f)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                _cache["metrics"] = json.load(f)

        # Combine train + test for analytics
        dfs = []
        for p in [TRAIN_PATH, TEST_PATH]:
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
        _cache["df"] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        _cache["loaded"] = True
        logger.info("All resources loaded successfully.")
    except Exception as e:
        logger.error(f"Resource loading failed: {e}")
        raise


_load()


# ─── Helpers ──────────────────────────────────────────────────
def _get(key):
    if not _cache.get("loaded"):
        _load()
    return _cache.get(key)


def _validate_required(data: dict, fields: list) -> str | None:
    """Returns error message if any required field is missing."""
    missing = [f for f in fields if f not in data or data[f] is None]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"
    return None


# ─── ENDPOINTS ────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict earthquake magnitude.
    Accepts JSON with: region, depth, year, month, day, hour, dayofweek
    Optional: latitude, longitude (auto-computed from region if missing)
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        err = _validate_required(data, ["region", "depth"])
        if err:
            return jsonify({"error": err}), 400

        model = _get("model")
        encoder = _get("encoder")
        feature_cols = _get("features")
        df = _get("df")

        # Build input row with defaults from training data medians
        row = {}
        for col in feature_cols:
            if col == "region":
                region_name = data["region"]
                if region_name not in encoder.classes_:
                    return jsonify({"error": f"Unknown region: {region_name}. "
                                             f"Valid: {list(encoder.classes_)}"}), 400
                row[col] = encoder.transform([region_name])[0]
            elif col in data:
                row[col] = float(data[col])
            elif col in ["latitude", "longitude"]:
                # Auto-fill from region center
                center = get_region_center(data["region"])
                row["latitude"] = center[0]
                row["longitude"] = center[1]
            else:
                # Use training median as fallback
                row[col] = float(df[col].median()) if col in df.columns else 0.0

        input_df = pd.DataFrame([row])[feature_cols]
        prediction = float(model.predict(input_df)[0])

        return jsonify({
            "magnitude": round(prediction, 2),
            "region": data["region"],
            "confidence": "high" if prediction < 6 else "moderate"
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/analytics/region", methods=["GET"])
def analytics_region():
    """Region-wise earthquake count and average magnitude."""
    try:
        df = _get("df")
        agg = (
            df.groupby("region")
            .agg(count=("magnitude", "size"),
                 avg_magnitude=("magnitude", "mean"),
                 max_magnitude=("magnitude", "max"),
                 avg_depth=("depth", "mean"))
            .reset_index()
        )
        agg["avg_magnitude"] = agg["avg_magnitude"].round(2)
        agg["max_magnitude"] = agg["max_magnitude"].round(2)
        agg["avg_depth"] = agg["avg_depth"].round(1)
        return jsonify(agg.sort_values("count", ascending=False).to_dict(orient="records"))
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/analytics/time", methods=["GET"])
def analytics_time():
    """Monthly earthquake frequency trends."""
    try:
        df = _get("df")
        agg = df.groupby(["year", "month"]).agg(
            count=("magnitude", "size"),
            avg_magnitude=("magnitude", "mean")
        ).reset_index()
        agg["time"] = agg["year"].astype(str) + "-" + agg["month"].astype(str).str.zfill(2)
        agg["avg_magnitude"] = agg["avg_magnitude"].round(2)
        agg = agg[["time", "count", "avg_magnitude"]].sort_values("time")
        return jsonify(agg.to_dict(orient="records"))
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/analytics/anomaly", methods=["GET"])
def analytics_anomaly():
    """
    Detect anomalous earthquakes using Z-score on magnitude.
    Returns events with |z-score| > 2.
    """
    try:
        df = _get("df")
        mag_mean = df["magnitude"].mean()
        mag_std = df["magnitude"].std()

        if mag_std == 0:
            return jsonify([])

        df_copy = df.copy()
        df_copy["z_score"] = (df_copy["magnitude"] - mag_mean) / mag_std
        anomalies = df_copy[df_copy["z_score"].abs() > 2].copy()

        result = anomalies[["latitude", "longitude", "depth", "magnitude",
                            "region", "origin_time", "z_score"]].copy()
        result["z_score"] = result["z_score"].round(2)
        result = result.sort_values("z_score", ascending=False)
        return jsonify(result.head(50).to_dict(orient="records"))
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/data", methods=["GET"])
def get_data():
    """
    Return processed dataset. Optional query params:
      ?limit=100   (default 200)
      ?region=Northeast
    """
    try:
        df = _get("df")
        limit = request.args.get("limit", 200, type=int)
        region = request.args.get("region", None)

        result = df.copy()
        if region:
            result = result[result["region"] == region]

        result = result.tail(limit)
        return jsonify(result.to_dict(orient="records"))
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/model/metrics", methods=["GET"])
def model_metrics():
    """Return model evaluation metrics."""
    try:
        metrics = _get("metrics")
        return jsonify(metrics if metrics else {"error": "No metrics found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/regions", methods=["GET"])
def list_regions():
    """Return list of valid region names."""
    try:
        encoder = _get("encoder")
        return jsonify(sorted(list(encoder.classes_)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _cache.get("loaded", False)})


# ─── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
