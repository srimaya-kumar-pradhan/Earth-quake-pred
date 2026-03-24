"""
Region mapping utility for Indian earthquake data.
Maps latitude/longitude pairs to one of 10 predefined Indian seismic zones.
Uses coordinate-based classification for performance (no API calls needed).
"""

# 10 Indian seismic regions defined by lat/lon bounding boxes.
# Order matters: more specific regions are checked first.
REGION_DEFINITIONS = [
    {
        "name": "Himalayan Belt",
        "lat_min": 30.0, "lat_max": 42.0,
        "lon_min": 72.0, "lon_max": 82.0,
        "description": "Himalayas, J&K, Ladakh, HP, Uttarakhand, Tibet border"
    },
    {
        "name": "Northeast",
        "lat_min": 21.0, "lat_max": 30.0,
        "lon_min": 89.0, "lon_max": 98.0,
        "description": "Assam, Meghalaya, Manipur, Mizoram, Nagaland, Tripura, Arunachal Pradesh"
    },
    {
        "name": "North",
        "lat_min": 26.0, "lat_max": 35.0,
        "lon_min": 74.0, "lon_max": 89.0,
        "description": "Delhi, UP, Bihar, Punjab, Haryana, Rajasthan (north)"
    },
    {
        "name": "Northwest",
        "lat_min": 22.0, "lat_max": 35.0,
        "lon_min": 64.0, "lon_max": 74.0,
        "description": "Gujarat, Rajasthan (west), Pakistan border, Afghanistan"
    },
    {
        "name": "Central",
        "lat_min": 20.0, "lat_max": 26.0,
        "lon_min": 74.0, "lon_max": 85.0,
        "description": "MP, Chhattisgarh, Jharkhand, central India"
    },
    {
        "name": "West",
        "lat_min": 15.0, "lat_max": 22.0,
        "lon_min": 68.0, "lon_max": 76.0,
        "description": "Maharashtra, Goa, coastal Karnataka"
    },
    {
        "name": "East",
        "lat_min": 18.0, "lat_max": 26.0,
        "lon_min": 82.0, "lon_max": 89.0,
        "description": "Odisha, West Bengal, eastern Bihar"
    },
    {
        "name": "Southwest",
        "lat_min": 8.0, "lat_max": 15.0,
        "lon_min": 72.0, "lon_max": 78.0,
        "description": "Kerala, south Karnataka, Lakshadweep"
    },
    {
        "name": "Southeast",
        "lat_min": 8.0, "lat_max": 15.0,
        "lon_min": 78.0, "lon_max": 85.0,
        "description": "Tamil Nadu, AP, Telangana (south)"
    },
    {
        "name": "South",
        "lat_min": 15.0, "lat_max": 20.0,
        "lon_min": 76.0, "lon_max": 82.0,
        "description": "Telangana, AP, northern Karnataka"
    },
]

# Extended regions for data outside India's mainland
EXTENDED_REGIONS = [
    {
        "name": "Andaman & Nicobar",
        "lat_min": 0.0, "lat_max": 16.0,
        "lon_min": 90.0, "lon_max": 100.0,
    },
    {
        "name": "Central Asia",
        "lat_min": 35.0, "lat_max": 45.0,
        "lon_min": 60.0, "lon_max": 80.0,
    },
    {
        "name": "Myanmar Region",
        "lat_min": 15.0, "lat_max": 28.0,
        "lon_min": 93.0, "lon_max": 100.0,
    },
]

ALL_REGIONS = REGION_DEFINITIONS + EXTENDED_REGIONS

# Unique region names for encoding and dropdowns
REGION_NAMES = sorted(list(set(r["name"] for r in ALL_REGIONS)))


def classify_region(lat: float, lon: float) -> str:
    """
    Classify a lat/lon coordinate into one of the predefined regions.
    Returns region name string.
    """
    for region in ALL_REGIONS:
        if (region["lat_min"] <= lat <= region["lat_max"] and
                region["lon_min"] <= lon <= region["lon_max"]):
            return region["name"]
    return "Other"


def get_region_center(region_name: str) -> tuple:
    """Returns approximate center (lat, lon) for a region."""
    for region in ALL_REGIONS:
        if region["name"] == region_name:
            lat_c = (region["lat_min"] + region["lat_max"]) / 2
            lon_c = (region["lon_min"] + region["lon_max"]) / 2
            return lat_c, lon_c
    return 20.5, 78.9  # India center fallback
