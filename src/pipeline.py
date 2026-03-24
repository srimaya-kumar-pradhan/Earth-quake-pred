"""
Production Data & Model Pipeline
=================================
Phase 1: Data preprocessing, region engineering, feature engineering
Phase 2: Time-series safe train/test split
Phase 3: XGBoost model training, evaluation, serialization
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
import warnings
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to path so we can import utils
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)
from core.regions import classify_region, REGION_NAMES

warnings.filterwarnings("ignore", category=UserWarning)

# ─── CONSTANTS ────────────────────────────────────────────────
INPUT_CSV = os.path.join(ROOT_DIR, "data", "raw", "Indian_earthquake_data.csv")
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "region_encoder.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


# ─── PHASE 1: DATA LOADING & PREPROCESSING ───────────────────
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """
    Load raw CSV, normalize columns, parse dates, clean numerics.
    """
    print("[Phase 1] Loading data...")
    df = pd.read_csv(csv_path)

    # Normalize column names: lowercase, strip whitespace
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Parse datetime — strip IST suffix to avoid timezone warnings
    df["origin_time"] = pd.to_datetime(
        df["origin_time"].str.replace(" IST", "", regex=False),
        format="mixed",
        dayfirst=False,
    )

    # Sort chronologically for time-series safety
    df = df.sort_values("origin_time").reset_index(drop=True)

    # Coerce numeric columns
    for col in ["depth", "magnitude", "latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaN numerics with column median
    df["depth"] = df["depth"].fillna(df["depth"].median())
    df["magnitude"] = df["magnitude"].fillna(df["magnitude"].median())
    df["latitude"] = df["latitude"].fillna(df["latitude"].median())
    df["longitude"] = df["longitude"].fillna(df["longitude"].median())

    print(f"  Loaded {len(df)} records, date range: "
          f"{df['origin_time'].min()} → {df['origin_time'].max()}")
    return df


# ─── PHASE 1b: REGION ENGINEERING ────────────────────────────
def add_region_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map lat/lon → named region, compute region-level aggregations.
    Replaces old lat_bin/lon_bin with meaningful region labels.
    """
    print("[Phase 1] Engineering region features...")

    # Vectorized region classification
    df["region"] = df.apply(
        lambda row: classify_region(row["latitude"], row["longitude"]), axis=1
    )

    # Cumulative region earthquake count (avoid future leakage)
    df["region_eq_count"] = df.groupby("region").cumcount() + 1

    # Running average magnitude per region (shifted to prevent leakage)
    df["region_avg_mag"] = (
        df.groupby("region")["magnitude"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["region_avg_mag"] = df["region_avg_mag"].fillna(df["magnitude"].mean())

    print(f"  Regions found: {df['region'].nunique()} → {df['region'].value_counts().to_dict()}")
    return df


# ─── PHASE 1c: FEATURE ENGINEERING ───────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from origin_time."""
    print("[Phase 1] Engineering temporal features...")
    df["year"] = df["origin_time"].dt.year
    df["month"] = df["origin_time"].dt.month
    df["day"] = df["origin_time"].dt.day
    df["hour"] = df["origin_time"].dt.hour
    df["dayofweek"] = df["origin_time"].dt.dayofweek
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling statistics on magnitude and depth.
    Uses shift(1) to prevent data leakage from current row.
    """
    print("[Phase 1] Engineering rolling features...")
    mag_shifted = df["magnitude"].shift(1)
    depth_shifted = df["depth"].shift(1)

    df["mag_roll_7"] = mag_shifted.rolling(window=7, min_periods=1).mean()
    df["mag_roll_30"] = mag_shifted.rolling(window=30, min_periods=1).mean()
    df["depth_roll_7"] = depth_shifted.rolling(window=7, min_periods=1).mean()

    # Fill remaining NaN from first few rows
    df["mag_roll_7"] = df["mag_roll_7"].fillna(df["magnitude"].mean())
    df["mag_roll_30"] = df["mag_roll_30"].fillna(df["magnitude"].mean())
    df["depth_roll_7"] = df["depth_roll_7"].fillna(df["depth"].mean())
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute energy release and inter-event time."""
    print("[Phase 1] Engineering derived features...")

    # Time since last earthquake (seconds)
    df["time_since_last"] = df["origin_time"].diff().dt.total_seconds().fillna(0)

    # Energy release: log-scale to prevent overflow (Gutenberg-Richter)
    # log10(E) = 1.5*M + 4.8 → we store log10(E) directly for numerical stability
    df["energy_release"] = 1.5 * df["magnitude"] + 4.8
    return df


# ─── PHASE 2: SPLITTING ──────────────────────────────────────
def split_and_save(df: pd.DataFrame) -> tuple:
    """
    Time-series safe 80/20 split. Data must already be sorted by origin_time.
    Saves train.csv and test.csv.
    """
    print("[Phase 2] Splitting dataset...")

    # Select columns for output
    keep_cols = [
        "latitude", "longitude", "depth",
        "year", "month", "day", "hour", "dayofweek",
        "mag_roll_7", "mag_roll_30", "depth_roll_7",
        "time_since_last", "energy_release",
        "region_eq_count", "region_avg_mag", "region",
        "magnitude",   # target
        "origin_time", "location",  # metadata (dropped before training)
    ]

    df_out = df[keep_cols].copy()

    # Final NaN check — drop any rows with NaN in numeric features
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    before = len(df_out)
    df_out = df_out.dropna(subset=numeric_cols)
    print(f"  Dropped {before - len(df_out)} rows with NaN values")

    split_idx = int(len(df_out) * 0.8)
    train = df_out.iloc[:split_idx].copy()
    test = df_out.iloc[split_idx:].copy()

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"  Train: {len(train)} rows → {TRAIN_PATH}")
    print(f"  Test:  {len(test)} rows → {TEST_PATH}")
    return train, test


# ─── PHASE 3: MODEL ──────────────────────────────────────────
def train_model(train: pd.DataFrame, test: pd.DataFrame):
    """
    Train XGBoost regressor, encode region, evaluate, serialize.
    """
    print("[Phase 3] Training model...")

    # Encode region
    le = LabelEncoder()
    all_regions = pd.concat([train["region"], test["region"]])
    le.fit(all_regions)

    train_enc = train.copy()
    test_enc = test.copy()
    train_enc["region"] = le.transform(train_enc["region"])
    test_enc["region"] = le.transform(test_enc["region"])

    # Drop non-numeric / metadata columns
    drop_cols = ["magnitude", "origin_time", "location"]
    feature_cols = [c for c in train_enc.columns if c not in drop_cols]

    X_train = train_enc[feature_cols].astype(float)
    y_train = train_enc["magnitude"]
    X_test = test_enc[feature_cols].astype(float)
    y_test = test_enc["magnitude"]

    # Train XGBoost
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    metrics = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
    print(f"  Metrics → RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # Serialize
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Model saved → {MODEL_PATH}")
    print(f"  Features saved → {FEATURES_PATH}")
    print(f"  Encoder saved → {LABEL_ENCODER_PATH}")
    print(f"  Metrics saved → {METRICS_PATH}")


# ─── MAIN ─────────────────────────────────────────────────────
def main():
    ensure_dirs()

    # Phase 1
    df = load_and_preprocess(INPUT_CSV)
    df = add_region_features(df)
    df = add_temporal_features(df)
    df = add_rolling_features(df)
    df = add_derived_features(df)

    # Phase 2
    train, test = split_and_save(df)

    # Phase 3
    train_model(train, test)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
