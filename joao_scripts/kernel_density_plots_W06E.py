# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:07:58 2026

@author: joaob
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import cv2
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import contextily as ctx
from pyproj import Transformer

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

BASE_DATA = Path(r"C:/Users/joaob/Dropbox/Documents/hackaton_UIC/hackaton_project/data")
BASE_OUT  = Path(r"C:/Users/joaob/Dropbox/Documents/hackaton_UIC/hackaton_project/data/output")

DATASETS = {
    "W06E_12-02-2025": {
        "jsonl_path"   : BASE_DATA / "W06E_12-02-2025/data/W06E_2025_12_2025-12-02.jsonl",
        "image_folder" : BASE_DATA / "W06E_12-02-2025/images",
    },
    "W042_08-09-2025": {
        "jsonl_path"   : BASE_DATA / "W042_08-09-2025/data/W042_2025_08_2025-08-09.jsonl",
        "image_folder" : BASE_DATA / "W042_08-09-2025/images",
    },
    "W065_08-17-2025": {
        "jsonl_path"   : BASE_DATA / "W065_08-17-2025/data/W065_2025_08_2025-08-17.jsonl",
        "image_folder" : BASE_DATA / "W065_08-17-2025/images",
    },
}

ACTIVE_DATASET = "W06E_12-02-2025"

_cfg          = DATASETS[ACTIVE_DATASET]
JSONL_PATH    = Path(_cfg["jsonl_path"])
IMAGE_FOLDER  = Path(_cfg["image_folder"])
SHP_PATH      = Path(r"C:\Users\joaob\Dropbox\Documents\hackaton_UIC\hackaton_project\data\SAGE_node_shapefiles\node_W06E.shp")

MODEL_NAMES   = ["YOLOv5n", "YOLOv8n", "YOLOv10n"]
LOCAL_TZ      = ZoneInfo("America/Chicago")
MODEL_TO_SHOW = "YOLOv8n"   # set to None to show all models

# Car detection filter — adjust class name to match your YOLO labels
CAR_CLASSES = {"car", "Car", "CAR", "vehicle", "Vehicle"}

OUT_MAIN = BASE_OUT / ACTIVE_DATASET
OUT_MAIN.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. TIME PERIODS
# =============================================================================

PERIOD_LEGEND = [
    ("Pre-dawn (0-6)",          "#4a90d9"),
    ("Morning twilight (7-8)",  "#e07b39"),
    ("Daylight (9-14)",         "#f5c842"),
    ("Dusk (15-16)",            "#c0392b"),
    ("Night (17-23)",           "#9b59b6"),
]

def get_period(hour: int) -> tuple[str, str]:
    if 0 <= hour <= 6:
        return PERIOD_LEGEND[0]
    elif 7 <= hour <= 8:
        return PERIOD_LEGEND[1]
    elif 9 <= hour <= 14:
        return PERIOD_LEGEND[2]
    elif 15 <= hour <= 16:
        return PERIOD_LEGEND[3]
    else:
        return PERIOD_LEGEND[4]

# =============================================================================
# 3. GROUND CONTROL POINTS
# =============================================================================

IMAGE_POINTS = np.float32([
  [1220, 890],   # GCP 1
  [819, 765],   # GCP 2
  [877, 843],  # GCP 3
  [2130, 1185],   # GCP 4
  [659, 1185],   # GCP 5
])

WORLD_POINTS = np.float32([
    [434756.062, 4648444.831],   # GCP 1
    [434756.471, 4648495.357],   # GCP 2
    [434749.445, 4648456.210],   # GCP 3
    [434756.651, 4648418.259],   # GCP 4
    [434738.281, 4648429.248],   # GCP 5
])

SHAPEFILE_EPSG = 26916

# =============================================================================
# 4. HELPERS
# =============================================================================

def parse_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                r["_line"] = line_no
                records.append(r)
            except json.JSONDecodeError:
                print(f"  [skip] invalid JSON at line {line_no}")
            if limit and len(records) >= limit:
                break
    return records

def parse_nested_json(value) -> dict | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None

def parse_timestamp(ts) -> datetime | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(
            int(ts) / 1_000_000_000, tz=timezone.utc
        ).astimezone(LOCAL_TZ)
    except (ValueError, TypeError):
        pass
    try:
        return datetime.fromisoformat(str(ts)).astimezone(LOCAL_TZ)
    except (ValueError, TypeError):
        return None

def get_centroid(bbox: list) -> list:
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]

# =============================================================================
# 5. HOMOGRAPHY
# =============================================================================

H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)
print("✓ Homography matrix computed")

def reproject_centroid(pixel_point, H):
    pt = np.float32([[pixel_point]])
    return cv2.perspectiveTransform(pt, H)[0][0]

# =============================================================================
# 6. PARSE + REPROJECT (cars only)
# =============================================================================

records = parse_jsonl(JSONL_PATH)
print(f"✓ Loaded {len(records)} records")

transformer_to_wgs84 = Transformer.from_crs(
    f"epsg:{SHAPEFILE_EPSG}", "epsg:4326", always_xy=True
)
transformer_to_webmercator = Transformer.from_crs(
    f"epsg:{SHAPEFILE_EPSG}", "epsg:3857", always_xy=True
)

reprojected = []
for rec in records:
    value = parse_nested_json(rec.get("value", rec))
    if value is None:
        continue

    ts_raw     = rec.get("timestamp") or rec.get("ts")
    local_time = parse_timestamp(ts_raw)
    if local_time is None:
        continue

    models_results = value.get("models_results", value)
    for model in MODEL_NAMES:
        if MODEL_TO_SHOW and model != MODEL_TO_SHOW:
            continue

        model_data = models_results.get(model, {})
        for det in model_data.get("detections", []):
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            obj_class = det.get("class", "")
            if obj_class not in CAR_CLASSES:
                continue

            centroid_px    = get_centroid(bbox)
            centroid_world = reproject_centroid(centroid_px, H)
            lon, lat       = transformer_to_wgs84.transform(
                centroid_world[0], centroid_world[1]
            )
            mx, my = transformer_to_webmercator.transform(
                centroid_world[0], centroid_world[1]
            )

            period_label, period_color = get_period(local_time.hour)

            reprojected.append({
                "timestamp"    : local_time,
                "hour"         : local_time.hour,
                "model"        : model,
                "class"        : obj_class,
                "confidence"   : det.get("confidence"),
                "lat"          : lat,
                "lon"          : lon,
                "mx"           : mx,
                "my"           : my,
                "period_label" : period_label,
                "period_color" : period_color,
            })

df = pd.DataFrame(reprojected)
if df.empty:
    raise ValueError("No car detections found! Check CAR_CLASSES labels against your data.")

df = df.sort_values("timestamp").reset_index(drop=True)
print(f"✓ Reprojected {len(df)} CAR detections across {df['timestamp'].nunique()} timestamps")
print(f"  Periods found: {df['period_label'].value_counts().to_dict()}")

# =============================================================================
# 7. SINGLE AGGREGATED HEXBIN MAP  +  Satellite basemap
# =============================================================================

# Web-Mercator bounding box with a small buffer
buf   = 30   # metres in EPSG:3857
xmin  = df["mx"].min() - buf
xmax  = df["mx"].max() + buf
ymin  = df["my"].min() - buf
ymax  = df["my"].max() + buf

# Make the extent square (required for undistorted hexagons)
cx    = (xmin + xmax) / 2
cy    = (ymin + ymax) / 2
half  = max(xmax - xmin, ymax - ymin) / 2 * 1.05   # 5 % margin
xmin, xmax = cx - half, cx + half
ymin, ymax = cy - half, cy + half

# Grid parameters
N_HEX      = 60      # approximate hexagon count across the shorter axis
EDGE_COLOR = "white"
EDGE_WIDTH = 0.5

# Single square figure
FIG_SIZE = 14   # inches — equal width and height

fig, ax = plt.subplots(1, 1, figsize=(FIG_SIZE, FIG_SIZE), facecolor="white")
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor("#cccccc")
    spine.set_linewidth(0.6)

# ── Satellite basemap — browser user-agent + multi-provider fallback ─────────
import requests, contextily.tile as ctx_tile

# Many tile CDNs (Esri, Google) block Python's default urllib user-agent.
# Monkey-patch the session used by contextily to mimic a real browser.
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.google.com/",
    "Accept-Language": "en-US,en;q=0.9",
})
try:
    ctx_tile._session = _SESSION   # contextily ≥ 1.3
except AttributeError:
    pass

BASEMAP_PROVIDERS = [
    ctx.providers.Esri.WorldImagery,
    ("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", "Google Satellite"),
    ctx.providers.OpenStreetMap.Mapnik,
]

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
basemap_loaded = False
for provider in BASEMAP_PROVIDERS:
    name = provider[1] if isinstance(provider, tuple) else provider.get("name", "?")
    src  = provider[0] if isinstance(provider, tuple) else provider
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=src,
            zoom="auto",
            attribution=False,
            reset_extent=False,
        )
        print(f"✓ Basemap loaded: {name}")
        basemap_loaded = True
        break
    except Exception as e:
        print(f"  [warn] provider failed ({name}): {e}")

if not basemap_loaded:
    ax.set_facecolor("#1a1a2e")
    print("  [warn] all basemap providers failed — using plain background")

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# ── Pre-compute mean confidence per hex cell ──────────────────────────────────
# Use hexbin with reduce_C_function=np.mean to colour by mean CI directly.
# mincnt=1 ensures empty cells are not drawn at all.
# Viridis-volcano palette: dark purple → teal → green → yellow → white-hot
cmap_hex = LinearSegmentedColormap.from_list(
    "viridis_volcano",
    [
        (0.267, 0.005, 0.329, 1.0),   # viridis dark purple
        (0.128, 0.566, 0.551, 1.0),   # viridis teal
        (0.369, 0.788, 0.383, 1.0),   # viridis green
        (0.993, 0.906, 0.144, 1.0),   # viridis yellow
        (1.000, 1.000, 0.900, 1.0),   # near-white hot tip
    ],
    N=256,
)
cmap_hex.set_under(alpha=0.0)   # cells with mincnt=0 → invisible

hb = ax.hexbin(
    df["mx"], df["my"],
    C=df["confidence"],
    reduce_C_function=np.mean,
    gridsize=N_HEX,
    extent=(xmin, xmax, ymin, ymax),
    cmap=cmap_hex,
    mincnt=1,
    alpha=1.0,
    linewidths=EDGE_WIDTH,
    edgecolors=EDGE_COLOR,
    zorder=2,
)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# ── Colorbar ──────────────────────────────────────────────────────────────────
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.08)
cb  = fig.colorbar(hb, cax=cax)
cb.set_label("Mean confidence score", fontsize=14, color="#333333")
cb.ax.yaxis.set_tick_params(color="#333333", labelsize=12)
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#333333")
cb.outline.set_edgecolor("#cccccc")

# ── Title & subtitle ──────────────────────────────────────────────────────────
ax.set_title(
    f"Car Detection Mean Confidence  ·  {ACTIVE_DATASET}  ·  Model: {MODEL_TO_SHOW}",
    fontsize=17, fontweight="bold", color="#111111", pad=12,
)
ax.text(
    0.5, -0.02,
    f"n = {len(df)} total car detections  ·  hexgrid N ≈ {N_HEX}  ·  colour = mean confidence score",
    transform=ax.transAxes,
    ha="center", va="top",
    color="#555555", fontsize=12,
)

plt.tight_layout(pad=0.8)

# =============================================================================
# 8. EXPORT
# =============================================================================

out_jpg = OUT_MAIN / f"hexgrid_cars_{ACTIVE_DATASET}_{MODEL_TO_SHOW}.jpg"
fig.savefig(
    out_jpg,
    dpi=250,
    format="jpeg",
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.close(fig)
print(f"✓ Map saved → {out_jpg}")