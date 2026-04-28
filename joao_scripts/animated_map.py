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
import folium
from folium.plugins import TimestampedGeoJson
import webbrowser
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

ACTIVE_DATASET = "W042_08-09-2025"

_cfg          = DATASETS[ACTIVE_DATASET]
JSONL_PATH    = Path(_cfg["jsonl_path"])
IMAGE_FOLDER  = Path(_cfg["image_folder"])
SHP_PATH      = Path(r"C:\Users\joaob\Dropbox\Documents\hackaton_UIC\hackaton_project\data\SAGE_node_shapefiles\node_W042.shp")

MODEL_NAMES   = ["YOLOv5n", "YOLOv8n", "YOLOv10n"]
LOCAL_TZ      = ZoneInfo("America/Chicago")
MODEL_TO_SHOW = "YOLOv8n"   # set to None to show all models

OUT_MAIN = BASE_OUT / ACTIVE_DATASET
OUT_MAIN.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. GROUND CONTROL POINTS
# =============================================================================

IMAGE_POINTS = np.float32([
    [910,  708],
    [1502, 708],
    [1921, 1724],
    [296,  1853],
])

WORLD_POINTS = np.float32([
    [445447.469, 4617438.348],
    [445442.892, 4617423.431],
    [445408.560, 4617435.365],
    [445409.454, 4617442.861],
])

SHAPEFILE_EPSG = 26916

# =============================================================================
# 3. HELPERS
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
# 4. HOMOGRAPHY
# =============================================================================

H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)
print("✓ Homography matrix computed")

def reproject_centroid(pixel_point, H):
    pt = np.float32([[pixel_point]])
    return cv2.perspectiveTransform(pt, H)[0][0]

# =============================================================================
# 5. PARSE + REPROJECT
# =============================================================================

records = parse_jsonl(JSONL_PATH, limit=100)
print(f"✓ Loaded {len(records)} records")

transformer = Transformer.from_crs(
    f"epsg:{SHAPEFILE_EPSG}", "epsg:4326", always_xy=True
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

            centroid_px    = get_centroid(bbox)
            centroid_world = reproject_centroid(centroid_px, H)
            lon, lat       = transformer.transform(
                centroid_world[0], centroid_world[1]
            )

            reprojected.append({
                "timestamp"  : local_time,
                "model"      : model,
                "class"      : det.get("class"),
                "confidence" : det.get("confidence"),
                "pixel_cx"   : centroid_px[0],
                "pixel_cy"   : centroid_px[1],
                "lat"        : lat,
                "lon"        : lon,
            })

df = pd.DataFrame(reprojected)
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"✓ Reprojected {len(df)} detections across {df['timestamp'].nunique()} timestamps")

# =============================================================================
# 6. BUILD TIMESTAMPED GEOJSON
# =============================================================================

CLASS_COLORS = {
    "car"        : "#3498db",
    "truck"      : "#e74c3c",
    "bus"        : "#e67e22",
    "person"     : "#2ecc71",
    "pedestrian" : "#2ecc71",
    "bicycle"    : "#9b59b6",
    "motorcycle" : "#1abc9c",
}

features = []
for _, row in df.iterrows():
    features.append({
        "type": "Feature",
        "geometry": {
            "type"       : "Point",
            "coordinates": [row["lon"], row["lat"]],
        },
        "properties": {
            "time"  : row["timestamp"].isoformat(),
            "popup" : (
                f"<b>{row['class']}</b><br>"
                f"Model: {row['model']}<br>"
                f"Conf: {row['confidence']:.2f}<br>"
                f"Time: {row['timestamp'].strftime('%H:%M:%S')}"
            ),
            "icon"  : "circle",
            "iconstyle": {
                "fillColor"   : CLASS_COLORS.get(row["class"], "#ffffff"),
                "fillOpacity" : 1.0,
                "stroke"      : True,
                "color"       : "#000000",   # ✅ black contour
                "opacity"     : 1.0,
                "weight"      : 2,           # ✅ contour thickness
                "radius"      : 6,
            },
        },
    })

geojson_data = {"type": "FeatureCollection", "features": features}

# =============================================================================
# 7. BUILD FOLIUM MAP
# =============================================================================

gdf = gpd.read_file(SHP_PATH).to_crs(epsg=4326)
center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]

m = folium.Map(
    location=center,
    zoom_start=20,
    max_zoom=21,
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google"
)

# ✅ 1. Fix black bar — inject full viewport CSS into the page <head>
fix_css = """
<style>
    html, body {
        margin:  0 !important;
        padding: 0 !important;
        height:  100% !important;
        width:   100% !important;
        overflow: hidden !important;
    }
    .folium-map {
        position: absolute !important;
        top:    0 !important;
        left:   0 !important;
        width:  100% !important;
        height: 100% !important;
    }
</style>
"""
m.get_root().header.add_child(folium.Element(fix_css))

# ✅ 2. Play bar stays at bottom — no relocation JS needed

# ✅ 3. No shapefile layer added

# Animated timestamped detections
TimestampedGeoJson(
    geojson_data,
    period="PT1S",
    duration="PT5S",
    auto_play=False,
    loop=True,
    max_speed=10,
    loop_button=True,
    date_options="YYYY-MM-DD HH:mm:ss",
    time_slider_drag_update=True,
    add_last_point=True,
).add_to(m)

# Legend
legend_html = """
<div style="
    position: fixed; bottom: 80px; left: 30px; z-index: 9999;
    background: rgba(0,0,0,0.85); color: white;
    font-family: 'Courier New', monospace; font-size: 13px;
    padding: 14px 18px; border-radius: 8px;
    border: 1px solid #555; line-height: 2;
">
    <b style="color:#ffff00; font-size:14px;">YOLO Detections</b><br>
    <span style="color:#3498db; font-size:18px;">●</span>&nbsp; Car<br>
    <span style="color:#e74c3c; font-size:18px;">●</span>&nbsp; Truck<br>
    <span style="color:#e67e22; font-size:18px;">●</span>&nbsp; Bus<br>
    <span style="color:#2ecc71; font-size:18px;">●</span>&nbsp; Person / Pedestrian<br>
    <span style="color:#9b59b6; font-size:18px;">●</span>&nbsp; Bicycle<br>
    <span style="color:#1abc9c; font-size:18px;">●</span>&nbsp; Motorcycle<br>
    <span style="color:#ffffff; font-size:18px;">●</span>&nbsp; Other<br>
    <hr style="border-color:#555; margin: 6px 0;">
    <span style="color:#aaaaaa; font-size:11px;">Click any point for details</span>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(m)

# =============================================================================
# 8. SAVE MAP
# =============================================================================

map_out = OUT_MAIN / f"{ACTIVE_DATASET}_animated_map.html"
m.save(str(map_out))
webbrowser.open(str(map_out))
print(f"✓ Animated map saved: {map_out}")