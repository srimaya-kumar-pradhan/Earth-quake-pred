"""
Production Data & Model Pipeline
=================================
Phase 1: Data loading (multi-format), preprocessing, region/feature engineering
Phase 2: Time-series safe train/test split
Phase 3: XGBoost model training, evaluation, serialization

Usage:
    # Default dataset (backward compatible):
    python src/pipeline.py

    # Custom dataset (CSV, JSON, XML, Excel):
    python src/pipeline.py --data data/raw/new_dataset.csv
    python src/pipeline.py --data data/raw/sample.xlsx
"""

import argparse
import json
import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ─── PATH SETUP ───────────────────────────────────────────────
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)

from core.regions import classify_region, REGION_NAMES
from data_loader import load_data

warnings.filterwarnings("ignore", category=UserWarning)

# ─── LOGGING ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── CONSTANTS ────────────────────────────────────────────────
DEFAULT_INPUT = os.path.join(ROOT_DIR, "data", "raw", "Indian_earthquake_data.csv")
DATA_DIR      = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR     = os.path.join(ROOT_DIR, "models")

TRAIN_PATH        = os.path.join(DATA_DIR, "train.csv")
TEST_PATH         = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH        = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH     = os.path.join(MODEL_DIR, "feature_columns.json")
LABEL_ENCODER_PATH= os.path.join(MODEL_DIR, "region_encoder.pkl")
METRICS_PATH      = os.path.join(MODEL_DIR, "metrics.json")


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


# ─── PHASE 1a: DATA LOADING ───────────────────────────────────

def load_and_preprocess(data_path: str) -> pd.DataFrame:
    """
    Load raw data from any supported format (CSV/JSON/XML/Excel),
    normalize schema, parse dates, coerce numerics.
    """
    logger.info(f"[Phase 1] Loading data from: {data_path}")

    # Universal loader — handles CSV, JSON, XML, Excel
    df = load_data(data_path)

    # Ensure origin_time column exists (may be named differently post-normalize)
    if "origin_time" not in df.columns:
        # Try to construct from date/time parts if available
        possible_time_cols = [c for c in df.columns if any(
            k in c for k in ["time", "date", "datetime", "timestamp"]
        )]
        if possible_time_cols:
            df["origin_time"] = pd.to_datetime(df[possible_time_cols[0]], errors="coerce")
            logger.info(f"[Phase 1] Mapped '{possible_time_cols[0]}' → origin_time")
        else:
            logger.warning("[Phase 1] No time column found; using synthetic timestamps.")
            df["origin_time"] = pd.date_range("2000-01-01", periods=len(df), freq="h")

    # Ensure origin_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["origin_time"]):
        df["origin_time"] = pd.to_datetime(
            df["origin_time"].astype(str).str.replace(" IST", "", regex=False),
            format="mixed",
            dayfirst=False,
            errors="coerce",
        )

    # Sort chronologically for time-series safety
    df = df.sort_values("origin_time").reset_index(drop=True)

    # Coerce required numeric columns
    for col in ["depth", "magnitude", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            logger.warning(f"[Phase 1] Column '{col}' missing — filling with 0.")
            df[col] = 0.0

    # Validate magnitude range
    mag_range = (df["magnitude"].min(), df["magnitude"].max())
    logger.info(f"[Phase 1] Magnitude range: {mag_range[0]:.1f} – {mag_range[1]:.1f}")

    # Drop rows with clearly invalid magnitudes (< 0 or > 10)
    before = len(df)
    df = df[(df["magnitude"] > 0) & (df["magnitude"] <= 10)].reset_index(drop=True)
    if len(df) < before:
        logger.info(f"[Phase 1] Dropped {before - len(df)} rows with invalid magnitudes")

    logger.info(
        f"[Phase 1] Loaded {len(df)} records, date range: "
        f"{df['origin_time'].min()} → {df['origin_time'].max()}"
    )
    return df


# ─── PHASE 1b: REGION ENGINEERING ────────────────────────────

def add_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map lat/lon → named region, compute region-level aggregations.
    """
    logger.info("[Phase 1] Engineering region features...")

    df["region"] = df.apply(
        lambda row: classify_region(row["latitude"], row["longitude"]), axis=1
    )

    # Cumulative region earthquake count (no future leakage)
    df["region_eq_count"] = df.groupby("region").cumcount() + 1

    # Running average magnitude per region (shifted to prevent leakage)
    df["region_avg_mag"] = (
        df.groupby("region")["magnitude"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["region_avg_mag"] = df["region_avg_mag"].fillna(df["magnitude"].mean())

    logger.info(f"[Phase 1] Regions found: {df['region'].nunique()} → {df['region'].value_counts().to_dict()}")
    return df


# ─── PHASE 1c: TEMPORAL FEATURES ─────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from origin_time, including cyclical encoding."""
    logger.info("[Phase 1] Engineering temporal features...")
    df["year"]      = df["origin_time"].dt.year
    df["month"]     = df["origin_time"].dt.month
    df["day"]       = df["origin_time"].dt.day
    df["hour"]      = df["origin_time"].dt.hour
    df["dayofweek"] = df["origin_time"].dt.dayofweek

    # Cyclical encoding for periodic features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


# ─── PHASE 1d: ROLLING FEATURES ──────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling statistics on magnitude and depth.
    Uses shift(1) to prevent data leakage from current row.
    """
    logger.info("[Phase 1] Engineering rolling features...")
    mag_shifted   = df["magnitude"].shift(1)
    depth_shifted = df["depth"].shift(1)

    df["mag_roll_7"]   = mag_shifted.rolling(window=7, min_periods=1).mean()
    df["mag_roll_30"]  = mag_shifted.rolling(window=30, min_periods=1).mean()
    df["depth_roll_7"] = depth_shifted.rolling(window=7, min_periods=1).mean()

    df["mag_roll_7"]   = df["mag_roll_7"].fillna(df["magnitude"].mean())
    df["mag_roll_30"]  = df["mag_roll_30"].fillna(df["magnitude"].mean())
    df["depth_roll_7"] = df["depth_roll_7"].fillna(df["depth"].mean())
    return df


# ─── PHASE 1e: DERIVED FEATURES ──────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute non-leaking derived features."""
    logger.info("[Phase 1] Engineering derived features...")

    df["time_since_last"] = df["origin_time"].diff().dt.total_seconds().fillna(0)

    # Depth-based seismic energy proxy (no magnitude leakage)
    # Deeper earthquakes in subduction zones tend to release more energy
    df["depth_log"] = np.log1p(df["depth"])

    # Depth categories (shallow vs deep earthquakes behave differently)
    df["is_shallow"] = (df["depth"] <= 10).astype(int)
    df["is_deep"]    = (df["depth"] >= 100).astype(int)

    # Latitude-longitude interaction (geospatial pattern)
    df["lat_lon_interaction"] = df["latitude"] * df["longitude"] / 1000.0

    # Magnitude standard deviation in rolling window (from shifted values)
    mag_shifted = df["magnitude"].shift(1)
    df["mag_std_7"] = mag_shifted.rolling(window=7, min_periods=1).std().fillna(0)

    return df


# ─── PHASE 2: SPLITTING ───────────────────────────────────────

def split_and_save(df: pd.DataFrame) -> tuple:
    """
    Time-series safe 80/20 split. Data must already be sorted by origin_time.
    Saves train.csv and test.csv.
    """
    logger.info("[Phase 2] Splitting dataset...")

    keep_cols = [
        "latitude", "longitude", "depth",
        "year", "month", "day", "hour", "dayofweek",
        "month_sin", "month_cos", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "mag_roll_7", "mag_roll_30", "depth_roll_7",
        "time_since_last", "depth_log", "is_shallow", "is_deep",
        "lat_lon_interaction", "mag_std_7",
        "region_eq_count", "region_avg_mag", "region",
        "magnitude",    # target
        "origin_time",  # metadata
    ]

    # Add 'location' only if it exists (backward compat)
    if "location" in df.columns:
        keep_cols.append("location")

    # Only keep columns that exist in df
    available = [c for c in keep_cols if c in df.columns]
    df_out = df[available].copy()

    # Final NaN check — drop rows with NaN in numeric features
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    before = len(df_out)
    df_out = df_out.dropna(subset=numeric_cols)
    logger.info(f"[Phase 2] Dropped {before - len(df_out)} rows with NaN values")

    split_idx = int(len(df_out) * 0.8)
    train = df_out.iloc[:split_idx].copy()
    test  = df_out.iloc[split_idx:].copy()

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    logger.info(f"[Phase 2] Train: {len(train)} rows → {TRAIN_PATH}")
    logger.info(f"[Phase 2] Test:  {len(test)} rows → {TEST_PATH}")

    # Log magnitude distribution in train/test
    logger.info(f"[Phase 2] Train mag range: {train['magnitude'].min():.2f} – {train['magnitude'].max():.2f}")
    logger.info(f"[Phase 2] Test mag range:  {test['magnitude'].min():.2f} – {test['magnitude'].max():.2f}")

    return train, test


# ─── PHASE 3: MODEL TRAINING ─────────────────────────────────

def train_model(train: pd.DataFrame, test: pd.DataFrame):
    """
    Train XGBoost regressor, encode region, evaluate, serialize.
    """
    logger.info("[Phase 3] Training model...")

    # Encode region
    le = LabelEncoder()
    all_regions = pd.concat([train["region"], test["region"]])
    le.fit(all_regions)

    train_enc = train.copy()
    test_enc  = test.copy()
    train_enc["region"] = le.transform(train_enc["region"])
    test_enc["region"]  = le.transform(test_enc["region"])

    # Drop non-numeric / metadata columns
    drop_cols = ["magnitude", "origin_time", "location"]
    feature_cols = [c for c in train_enc.columns if c not in drop_cols]

    X_train = train_enc[feature_cols].astype(float)
    y_train = train_enc["magnitude"]
    X_test  = test_enc[feature_cols].astype(float)
    y_test  = test_enc["magnitude"]

    logger.info(f"[Phase 3] Features ({len(feature_cols)}): {feature_cols}")
    logger.info(f"[Phase 3] X_train shape: {X_train.shape}, y_train range: {y_train.min():.2f}–{y_train.max():.2f}")

    # Train XGBoost with tuned hyperparameters (from RandomizedSearchCV)
    model = XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    preds = model.predict(X_test)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae   = float(mean_absolute_error(y_test, preds))
    r2    = float(r2_score(y_test, preds))

    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
    logger.info(f"[Phase 3] Metrics → RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # Prediction sanity check
    pred_range = (preds.min(), preds.max())
    logger.info(f"[Phase 3] Prediction range: {pred_range[0]:.2f} – {pred_range[1]:.2f}")

    # Feature importance (top 10)
    importances = dict(zip(feature_cols, model.feature_importances_))
    top10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info(f"[Phase 3] Top features: {top10}")

    # Serialize
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"[Phase 3] Model saved → {MODEL_PATH}")
    logger.info(f"[Phase 3] Features saved → {FEATURES_PATH}")
    logger.info(f"[Phase 3] Encoder saved → {LABEL_ENCODER_PATH}")
    logger.info(f"[Phase 3] Metrics saved → {METRICS_PATH}")


# ─── CLI + MAIN ───────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Earthquake ML Pipeline — train XGBoost on any dataset format"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Path to input dataset (CSV, JSON, XML, or Excel). "
            f"Defaults to '{DEFAULT_INPUT}'."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data if args.data else DEFAULT_INPUT

    # Validate input
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found: {data_path}")
        sys.exit(1)

    logger.info(f"━━━ Using dataset: {data_path} ━━━")
    ensure_dirs()

    # Phase 1: Load + Engineer
    df = load_and_preprocess(data_path)
    df = add_region_features(df)
    df = add_temporal_features(df)
    df = add_rolling_features(df)
    df = add_derived_features(df)

    # Phase 2: Split
    train, test = split_and_save(df)

    # Phase 3: Train
    train_model(train, test)

    logger.info("✅ Pipeline complete.")


if __name__ == "__main__":
    main()
