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

ACTIVE_DATASET = "W042_08-09-2025"   # <-- change to switch dataset

_cfg         = DATASETS[ACTIVE_DATASET]
JSONL_PATH   = Path(_cfg["jsonl_path"])
IMAGE_FOLDER = Path(_cfg["image_folder"])

SHP_PATH     = Path(r"C:\Users\joaob\Dropbox\Documents\hackaton_UIC\hackaton_project\data\SAGE_node_shapefiles\node_W042.shp")

MODEL_NAMES  = ["YOLOv5n", "YOLOv8n", "YOLOv10n"]
LOCAL_TZ     = ZoneInfo("America/Chicago")

OUT_MAIN     = BASE_OUT / ACTIVE_DATASET
OUT_MAIN.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. GROUND CONTROL POINTS (GCPs)
# =============================================================================

IMAGE_POINTS = np.float32([
    [910,  708],    # GCP 1
    [1502, 708],    # GCP 2
    [1921, 1724],   # GCP 3
    [296,  1853],   # GCP 4
])

WORLD_POINTS = np.float32([
    [445447.469, 4617438.348],   # GCP 1
    [445442.892, 4617423.431],   # GCP 2
    [445408.560, 4617435.365],   # GCP 3
    [445409.454, 4617442.861],   # GCP 4
])

SHAPEFILE_EPSG = 26916   # UTM Zone 16N — confirmed

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
    """Handle both nanosecond integers AND ISO 8601 strings."""
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
    """[x1, y1, x2, y2] → [cx, cy]"""
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]

# =============================================================================
# 4. COMPUTE HOMOGRAPHY  (runs once)
# =============================================================================

H, mask = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)
print("✓ Homography matrix computed")
print(H)

def reproject_centroid(pixel_point: list, H: np.ndarray) -> np.ndarray:
    """Apply homography matrix to a pixel centroid → world (X, Y)."""
    pt = np.float32([[pixel_point]])
    projected = cv2.perspectiveTransform(pt, H)
    return projected[0][0]

# =============================================================================
# 5. PARSE JSONL + EXTRACT + REPROJECT ALL CENTROIDS
# =============================================================================

records = parse_jsonl(JSONL_PATH, limit=100)   # ✅ remove limit to process all
print(f"✓ Loaded {len(records)} records from JSONL")

reprojected = []

for rec in records:
    value = parse_nested_json(rec.get("value", rec))
    if value is None:
        continue

    ts_raw     = rec.get("timestamp") or rec.get("ts")
    local_time = parse_timestamp(ts_raw)

    models_results = value.get("models_results", value)
    for model in MODEL_NAMES:
        model_data = models_results.get(model, {})
        detections = model_data.get("detections", [])

        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            centroid_px    = get_centroid(bbox)
            centroid_world = reproject_centroid(centroid_px, H)

            reprojected.append({
                "timestamp"  : local_time,
                "model"      : model,
                "class"      : det.get("class"),
                "confidence" : det.get("confidence"),
                "pixel_cx"   : centroid_px[0],
                "pixel_cy"   : centroid_px[1],
                "world_x"    : float(centroid_world[0]),
                "world_y"    : float(centroid_world[1]),
                "bbox_x1"    : bbox[0],
                "bbox_y1"    : bbox[1],
                "bbox_x2"    : bbox[2],
                "bbox_y2"    : bbox[3],
            })

df_proj = pd.DataFrame(reprojected)
print(f"✓ Reprojected {len(df_proj)} detections")
print(df_proj.head())

# Save to CSV
csv_out = OUT_MAIN / f"{ACTIVE_DATASET}_reprojected.csv"
df_proj.to_csv(csv_out, index=False)
print(f"✓ CSV saved: {csv_out}")

# =============================================================================
# 6. PLOT ON INTERACTIVE MAP
# =============================================================================

gdf = gpd.read_file(SHP_PATH)
gdf = gdf.to_crs(epsg=4326)
center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]

transformer = Transformer.from_crs(
    f"epsg:{SHAPEFILE_EPSG}", "epsg:4326", always_xy=True
)

m = folium.Map(
    location=center,
    zoom_start=20,
    max_zoom=21,
    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
    attr="Google"
)

# ✅ Add this right after the map is created
basemap_css = """
<style>
    .leaflet-tile-pane {
        opacity: 1;   /* adjust this value to taste */
    }
</style>
"""
m.get_root().html.add_child(folium.Element(basemap_css))

# GIS layer
folium.GeoJson(
    gdf,
    name="Intersection (GIS)",
    style_function=lambda x: {
        "color": "#ffff00",
        "weight": 2,
        "fillOpacity": 0.1,
    }
).add_to(m)

# Color per object class
CLASS_COLORS = {
    "car"        : "#3498db",
    "truck"      : "#e74c3c",
    "bus"        : "#e67e22",
    "person"     : "#2ecc71",
    "pedestrian" : "#2ecc71",
    "bicycle"    : "#9b59b6",
    "motorcycle" : "#1abc9c",
}

# ✅ Plot each reprojected centroid with 40% transparency
skipped = 0
for _, row in df_proj.iterrows():
    try:
        lon, lat = transformer.transform(row["world_x"], row["world_y"])
        color = CLASS_COLORS.get(row["class"], "#ffffff")
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,       # ✅ 40% transparency
            opacity=0.8,            # ✅ border also semi-transparent
            tooltip=(
                f"<b>{row['class']}</b><br>"
                f"Model: {row['model']}<br>"
                f"Conf: {row['confidence']:.2f}<br>"
                f"Time: {row['timestamp']}<br>"
                f"Pixel: ({row['pixel_cx']:.0f}, {row['pixel_cy']:.0f})"
            )
        ).add_to(m)
    except Exception:
        skipped += 1
        continue

print(f"✓ Plotted {len(df_proj) - skipped} detections ({skipped} skipped)")

# ✅ Legend
legend_html = """
<div style="
    position: fixed; bottom: 30px; left: 30px; z-index: 9999;
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

map_out = OUT_MAIN / f"{ACTIVE_DATASET}_reprojected_map.html"
m.save(str(map_out))
webbrowser.open(str(map_out))
print(f"✓ Interactive map saved: {map_out}")