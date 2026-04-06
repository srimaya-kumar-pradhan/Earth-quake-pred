"""
Flask Backend — Earthquake ML API (Production-Ready)
=====================================================
Endpoints:
  POST /predict          → magnitude prediction
  GET  /analytics/region → per-region stats
  GET  /analytics/time   → monthly trends
  GET  /analytics/anomaly→ z-score anomalies
  GET  /data             → full processed dataset
  GET  /model/metrics    → model eval metrics
  GET  /regions          → list of all regions
  GET  /health           → health check
  GET  /                 → serves frontend dashboard

Run locally:
    python src/api/app.py

Production (Gunicorn):
    gunicorn -b 0.0.0.0:5000 "src.api.app:app"
"""

import json
import logging
import os
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ─── Path Setup ───────────────────────────────────────────────
# Support both: `python src/api/app.py` and `gunicorn src.api.app:app`
API_DIR  = os.path.dirname(os.path.abspath(__file__))   # src/api/
SRC_DIR  = os.path.dirname(API_DIR)                      # src/
ROOT_DIR = os.path.dirname(SRC_DIR)                      # project root
sys.path.insert(0, SRC_DIR)

from core.regions import REGION_NAMES, classify_region, get_region_center

# ─── Flask App ────────────────────────────────────────────────
# Serve static files from src/web/
WEB_DIR = os.path.join(SRC_DIR, "web")
app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")

# CORS — allow cross-origin requests from dev servers (Live Server, etc.)
# In production (Docker), frontend is served by Flask on same origin so CORS
# headers are harmless but unused. We allow common dev ports explicitly.
CORS(app, resources={r"/*": {
    "origins": [
        "http://127.0.0.1:5500",   # VS Code Live Server (default)
        "http://localhost:5500",
        "http://127.0.0.1:5501",   # Live Server alternate
        "http://localhost:5501",
        "http://127.0.0.1:3000",   # React/Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:8080",   # Generic dev
        "http://localhost:8080",
    ],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
}})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── File Paths ───────────────────────────────────────────────
MODEL_PATH    = os.path.join(ROOT_DIR, "models", "model.pkl")
FEATURES_PATH = os.path.join(ROOT_DIR, "models", "feature_columns.json")
ENCODER_PATH  = os.path.join(ROOT_DIR, "models", "region_encoder.pkl")
METRICS_PATH  = os.path.join(ROOT_DIR, "models", "metrics.json")
TRAIN_PATH    = os.path.join(ROOT_DIR, "data", "processed", "train.csv")
TEST_PATH     = os.path.join(ROOT_DIR, "data", "processed", "test.csv")

# ─── Resource Cache ───────────────────────────────────────────
_cache = {}


def _load():
    """Load model, encoder, features, and data into memory once."""
    if _cache.get("loaded"):
        return
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python src/pipeline.py` first to train the model."
            )

        _cache["model"]   = joblib.load(MODEL_PATH)
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
        logger.info("✅ All resources loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Resource loading failed: {e}")
        _cache["loaded"]  = False
        _cache["load_err"] = str(e)


# Load on startup
_load()


# ─── Helpers ──────────────────────────────────────────────────
def _get(key):
    if not _cache.get("loaded"):
        _load()
    return _cache.get(key)


def _validate_required(data: dict, fields: list):
    """Returns error message string if any required field is missing."""
    missing = [f for f in fields if f not in data or data[f] is None]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"
    return None


# ─── FRONTEND SERVING ─────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend dashboard."""
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve static assets (CSS, JS, etc.) — only for files that exist."""
    filepath = os.path.join(WEB_DIR, filename)
    if os.path.isfile(filepath):
        return send_from_directory(WEB_DIR, filename)
    # Not a static file → let Flask handle as 404 (or match another route)
    return jsonify({"error": "Not found"}), 404


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

        model        = _get("model")
        encoder      = _get("encoder")
        feature_cols = _get("features")
        df           = _get("df")

        if model is None:
            return jsonify({"error": _cache.get("load_err", "Model not loaded")}), 503

        # Build input row with defaults from training data medians
        row = {}
        for col in feature_cols:
            if col == "region":
                region_name = data["region"]
                if region_name not in encoder.classes_:
                    return jsonify({
                        "error": f"Unknown region: '{region_name}'. "
                                 f"Valid regions: {sorted(list(encoder.classes_))}"
                    }), 400
                row[col] = int(encoder.transform([region_name])[0])
            elif col in data:
                row[col] = float(data[col])
            elif col in ["latitude", "longitude"]:
                center = get_region_center(data["region"])
                row["latitude"]  = center[0]
                row["longitude"] = center[1]
            else:
                # Use training median as fallback
                row[col] = float(df[col].median()) if (df is not None and col in df.columns) else 0.0

        input_df   = pd.DataFrame([row])[feature_cols]
        prediction = float(model.predict(input_df)[0])

        return jsonify({
            "magnitude":  round(prediction, 2),
            "region":     data["region"],
            "confidence": "high" if prediction < 6 else "moderate",
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/analytics/region", methods=["GET"])
def analytics_region():
    """Region-wise earthquake count and average magnitude."""
    try:
        df = _get("df")
        if df is None or df.empty:
            return jsonify([])

        agg = (
            df.groupby("region")
            .agg(
                count=("magnitude", "size"),
                avg_magnitude=("magnitude", "mean"),
                max_magnitude=("magnitude", "max"),
                avg_depth=("depth", "mean"),
            )
            .reset_index()
        )
        agg["avg_magnitude"] = agg["avg_magnitude"].round(2)
        agg["max_magnitude"] = agg["max_magnitude"].round(2)
        agg["avg_depth"]     = agg["avg_depth"].round(1)
        return jsonify(agg.sort_values("count", ascending=False).to_dict(orient="records"))
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/analytics/time", methods=["GET"])
def analytics_time():
    """Monthly earthquake frequency trends."""
    try:
        df = _get("df")
        if df is None or df.empty:
            return jsonify([])

        agg = df.groupby(["year", "month"]).agg(
            count=("magnitude", "size"),
            avg_magnitude=("magnitude", "mean"),
        ).reset_index()
        agg["time"]          = agg["year"].astype(str) + "-" + agg["month"].astype(str).str.zfill(2)
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
        if df is None or df.empty:
            return jsonify([])

        mag_mean = df["magnitude"].mean()
        mag_std  = df["magnitude"].std()

        if mag_std == 0:
            return jsonify([])

        df_copy           = df.copy()
        df_copy["z_score"] = (df_copy["magnitude"] - mag_mean) / mag_std
        anomalies          = df_copy[df_copy["z_score"].abs() > 2].copy()

        # Only include columns that exist
        cols = ["latitude", "longitude", "depth", "magnitude", "region", "z_score"]
        if "origin_time" in anomalies.columns:
            cols.append("origin_time")

        result            = anomalies[cols].copy()
        result["z_score"] = result["z_score"].round(2)
        result            = result.sort_values("z_score", ascending=False)
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
        if df is None or df.empty:
            return jsonify([])

        limit  = request.args.get("limit", 200, type=int)
        region = request.args.get("region", None)

        result = df.copy()
        if region:
            result = result[result["region"] == region]

        result = result.tail(limit)
        # Replace NaN with None for JSON serialization
        return jsonify(result.where(pd.notnull(result), None).to_dict(orient="records"))
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
        if encoder is None:
            return jsonify({"error": _cache.get("load_err", "Encoder not loaded")}), 503
        return jsonify(sorted(list(encoder.classes_)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok" if _cache.get("loaded") else "degraded",
        "model_loaded": _cache.get("loaded", False),
        "error":        _cache.get("load_err"),
    })


# ─── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    logger.info(f"Starting Flask on port {port} (debug={debug})")
    app.run(debug=debug, host="0.0.0.0", port=port)
