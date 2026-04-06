"""
Universal Data Ingestion Layer — data_loader.py
================================================
Supports CSV, JSON, XML, Excel (.xlsx) formats.
Auto-detects format, parses, normalizes schema,
cleans data, and returns an ML-ready DataFrame.

Usage:
    from src.data_loader import load_data
    df = load_data("data/raw/sample.xml")
"""

import os
import logging
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─── Column Alias Mapping ─────────────────────────────────────
# Maps common alternate column names → canonical names
COLUMN_ALIASES = {
    # latitude
    "lat": "latitude",
    "LAT": "latitude",
    "Latitude": "latitude",
    "LATITUDE": "latitude",
    "lat.": "latitude",
    "y": "latitude",

    # longitude
    "lon": "longitude",
    "lng": "longitude",
    "long": "longitude",
    "LON": "longitude",
    "Longitude": "longitude",
    "LONGITUDE": "longitude",
    "lon.": "longitude",
    "x": "longitude",

    # magnitude
    "mag": "magnitude",
    "Magnitude": "magnitude",
    "MAGNITUDE": "magnitude",
    "richter": "magnitude",
    "ml": "magnitude",
    "Ml": "magnitude",
    "ML": "magnitude",
    "mb": "magnitude",
    "Ms": "magnitude",

    # depth
    "Depth": "depth",
    "DEPTH": "depth",
    "depth_km": "depth",
    "focal_depth": "depth",

    # time / date
    "time": "origin_time",
    "Time": "origin_time",
    "date": "origin_time",
    "Date": "origin_time",
    "datetime": "origin_time",
    "DateTime": "origin_time",
    "origin": "origin_time",
    "event_time": "origin_time",
    "timestamp": "origin_time",
    "UTC": "origin_time",
    "utc_datetime": "origin_time",
    "date_time": "origin_time",

    # location
    "place": "location",
    "Place": "location",
    "region": "location",
    "Region": "location",
    "Location": "location",
    "description": "location",
}

# Required columns for downstream pipeline compatibility
REQUIRED_COLUMNS = ["latitude", "longitude", "magnitude", "depth"]


# ─── FORMAT DETECTION ─────────────────────────────────────────

def detect_format(file_path: str) -> str:
    """
    Detect file format by extension.
    Returns one of: 'csv', 'json', 'xml', 'excel'
    Raises ValueError for unsupported formats.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    format_map = {
        ".csv": "csv",
        ".json": "json",
        ".xml": "xml",
        ".xlsx": "excel",
        ".xls": "excel",
    }
    if ext not in format_map:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported: {list(format_map.keys())}"
        )
    return format_map[ext]


# ─── PARSERS ──────────────────────────────────────────────────

def _parse_csv(file_path: str) -> pd.DataFrame:
    """Parse a CSV file into a DataFrame."""
    logger.info(f"[DataLoader] Parsing CSV: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
    return df


def _parse_json(file_path: str) -> pd.DataFrame:
    """Parse a JSON file. Supports flat, records, and GeoJSON-like structures."""
    logger.info(f"[DataLoader] Parsing JSON: {file_path}")
    import json

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # GeoJSON FeatureCollection
    if isinstance(raw, dict) and raw.get("type") == "FeatureCollection":
        records = []
        for feat in raw.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", []) if geom else []
            if len(coords) >= 2:
                props["longitude"] = coords[0]
                props["latitude"] = coords[1]
                if len(coords) >= 3:
                    props.setdefault("depth", coords[2])
            records.append(props)
        return pd.DataFrame(records)

    # Generic dict wrapping a list
    if isinstance(raw, dict):
        for key in raw:
            if isinstance(raw[key], list):
                return pd.json_normalize(raw[key])
        return pd.json_normalize([raw])

    # Already a list of records
    if isinstance(raw, list):
        return pd.json_normalize(raw)

    raise ValueError("Cannot parse JSON: unrecognized structure.")


def parse_xml(file_path: str) -> pd.DataFrame:
    """
    Parse an XML file into a DataFrame.
    Each child element of the root is considered one record.
    Sub-elements of each record become columns.
    """
    logger.info(f"[DataLoader] Parsing XML: {file_path}")
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Corrupted XML file: {e}")

    records = []
    for child in root:
        row = {}
        for elem in child:
            tag = elem.tag.split("}")[-1]  # strip namespace
            row[tag] = elem.text
        # Also capture attributes
        row.update(child.attrib)
        records.append(row)

    if not records:
        raise ValueError("XML file is empty or has no parseable records.")
    return pd.DataFrame(records)


def _parse_excel(file_path: str) -> pd.DataFrame:
    """Parse an Excel (.xlsx/.xls) file into a DataFrame."""
    logger.info(f"[DataLoader] Parsing Excel: {file_path}")
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception:
        df = pd.read_excel(file_path, engine="xlrd")
    return df


# ─── SCHEMA NORMALIZATION ─────────────────────────────────────

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to canonical names using COLUMN_ALIASES.
    Performs case-insensitive fuzzy matching on column names.
    """
    # Build a lowercase → canonical mapping
    alias_lower = {k.lower(): v for k, v in COLUMN_ALIASES.items()}

    rename_map = {}
    for col in df.columns:
        stripped = col.strip()
        # Direct alias match
        if stripped in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[stripped]
        # Case-insensitive alias match
        elif stripped.lower() in alias_lower:
            rename_map[col] = alias_lower[stripped.lower()]

    if rename_map:
        logger.info(f"[DataLoader] Renaming columns: {rename_map}")
        df = df.rename(columns=rename_map)

    # Lowercase and strip all remaining column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    return df


# ─── DATA CLEANING ────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and type-cast the DataFrame for ML readiness:
    - Coerce numeric columns
    - Parse timestamps
    - Drop duplicates and rows with all-NaN
    - Fill numeric NaN with median
    - Validate required columns
    """
    logger.info(f"[DataLoader] Cleaning dataset ({len(df)} rows)...")

    # Drop fully-empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        logger.info(f"[DataLoader] Removed {before - len(df)} duplicate rows.")

    # ── Numeric coercion ──
    numeric_cols = ["latitude", "longitude", "magnitude", "depth"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Timestamp parsing for origin_time ──
    if "origin_time" in df.columns:
        df["origin_time"] = df["origin_time"].astype(str).str.replace(" IST", "", regex=False)
        df["origin_time"] = pd.to_datetime(df["origin_time"], format="mixed", dayfirst=False, errors="coerce")
        df = df.sort_values("origin_time").reset_index(drop=True)

    # ── Fill NaNs with column median for numerics ──
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            med = df[col].median()
            fills = df[col].isna().sum()
            df[col] = df[col].fillna(med)
            logger.info(f"[DataLoader] Filled {fills} NaN in '{col}' with median={med:.3f}")

    # ── Validate required columns (warn, don't crash) ──
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        logger.warning(
            f"[DataLoader] ⚠ Missing required columns: {missing_required}. "
            f"Pipeline may fail downstream."
        )

    logger.info(f"[DataLoader] Clean dataset: {len(df)} rows, columns: {list(df.columns)}")
    return df


# ─── PUBLIC API ───────────────────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame:
    """
    Universal data loader. Accepts CSV, JSON, XML, or Excel.
    Returns a clean, normalized DataFrame ready for ML pipeline.

    Args:
        file_path: Absolute or relative path to the data file.

    Returns:
        pd.DataFrame: Cleaned and schema-normalized DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If format is unsupported or file is corrupted.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    fmt = detect_format(file_path)
    logger.info(f"[DataLoader] Detected format: {fmt.upper()} → {file_path}")

    parser_map = {
        "csv": _parse_csv,
        "json": _parse_json,
        "xml": parse_xml,
        "excel": _parse_excel,
    }

    df_raw = parser_map[fmt](file_path)

    if df_raw is None or df_raw.empty:
        raise ValueError(f"Loaded data is empty: {file_path}")

    df_norm = normalize_schema(df_raw)
    df_clean = clean_data(df_norm)

    return df_clean


# ─── EXAMPLE USAGE ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Indian_earthquake_data.csv"
    df = load_data(path)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
