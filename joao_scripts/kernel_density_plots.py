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
from scipy.stats import gaussian_kde
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

ACTIVE_DATASET = "W042_08-09-2025"

_cfg          = DATASETS[ACTIVE_DATASET]
JSONL_PATH    = Path(_cfg["jsonl_path"])
IMAGE_FOLDER  = Path(_cfg["image_folder"])
SHP_PATH      = Path(r"C:\Users\joaob\Dropbox\Documents\hackaton_UIC\hackaton_project\data\SAGE_node_shapefiles\node_W042.shp")

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
# 7. KDE MAP — one subplot per period  +  Google Satellite basemap
# =============================================================================

periods_present = [p for p in PERIOD_LEGEND if p[0] in df["period_label"].unique()]
n_periods = len(periods_present)

# 2-row grid layout
ncols   = 3
nrows   = -(-n_periods // ncols)   # ceiling division
fig_w   = 6.5 * ncols + 0.5
fig_h   = 4.2 * nrows + 1.2

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(fig_w, fig_h),
    facecolor="white",
)
axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

# Hide unused cells
for idx in range(n_periods, len(axes_flat)):
    axes_flat[idx].set_visible(False)

# Web-Mercator bounding box with a small buffer
buf   = 30   # metres in EPSG:3857 ≈ ~30 m
xmin  = df["mx"].min() - buf
xmax  = df["mx"].max() + buf
ymin  = df["my"].min() - buf
ymax  = df["my"].max() + buf

kde_max_global = 0  # for a shared color scale

kde_results = {}
for label, color in periods_present:
    sub = df[df["period_label"] == label]
    if len(sub) < 2:
        kde_results[label] = None
        continue
    xy  = np.vstack([sub["mx"], sub["my"]])
    kde = gaussian_kde(xy, bw_method="scott")

    # Evaluate on a fine grid
    xi  = np.linspace(xmin, xmax, 300)
    yi  = np.linspace(ymin, ymax, 300)
    XX, YY = np.meshgrid(xi, yi)
    ZZ  = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    kde_results[label] = (xi, yi, ZZ)
    kde_max_global = max(kde_max_global, ZZ.max())

# Build a colormap: transparent at low density, opaque at high density,
# with a power-law gamma to crush low-density areas and boost contrast.
def period_cmap(hex_color, gamma=3.0):
    """
    gamma > 1 → transparency ramps up steeply, low-density regions nearly
    invisible, high-density hotspots vivid.  Increase gamma for more contrast.
    """
    r, g, b = mcolors.to_rgb(hex_color)
    n = 256
    alphas = np.linspace(0, 1, n) ** gamma          # power-law alpha ramp
    colors = [(r, g, b, float(a)) for a in alphas]
    return LinearSegmentedColormap.from_list("custom", colors, N=n)

for ax, (label, color) in zip(axes_flat, periods_present):
    ax.set_facecolor("white")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.6)

    # ── Google Satellite basemap ──────────────────────────────────────────
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.Esri.WorldImagery,
            zoom="auto",
            attribution=False,
        )
    except Exception as e:
        print(f"  [warn] basemap failed for '{label}': {e}")

    # ── KDE overlay ──────────────────────────────────────────────────────
    res = kde_results.get(label)
    if res is not None:
        xi, yi, ZZ = res

        # Normalise per-panel so the hottest cell = 1.0,
        # then apply power-law: values below ~50% of peak become near-transparent.
        ZZ_norm = ZZ / ZZ.max() if ZZ.max() > 0 else ZZ
        ZZ_powered = ZZ_norm ** 2.5          # squash low-density floor

        cmap = period_cmap(color, gamma=3.0)
        ax.pcolormesh(
            xi, yi, ZZ_powered,
            cmap=cmap,
            vmin=0, vmax=1,
            shading="gouraud",
            zorder=2,
        )

    # ── Scatter dots ─────────────────────────────────────────────────────
    sub = df[df["period_label"] == label]
    ax.scatter(
        sub["mx"], sub["my"],
        s=6, c=color, alpha=0.35, linewidths=0,
        zorder=3,
    )

    # ── Period title ─────────────────────────────────────────────────────
    ax.set_title(
        label, color=color,
        fontsize=11, fontweight="bold",
        pad=7,
    )
    ax.text(
        0.5, -0.03,
        f"n = {len(sub)} detections",
        transform=ax.transAxes,
        ha="center", va="top",
        color="#555555", fontsize=9.5,
    )

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    Patch(facecolor=c, label=lbl)
    for lbl, c in PERIOD_LEGEND
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(PERIOD_LEGEND),
    frameon=False,
    labelcolor="#222222",
    fontsize=10,
    bbox_to_anchor=(0.5, -0.01),
)

# ── Main title ────────────────────────────────────────────────────────────────
fig.suptitle(
    f"Car Detection Density  ·  {ACTIVE_DATASET}  ·  Model: {MODEL_TO_SHOW}",
    color="#111111", fontsize=14, fontweight="bold", y=1.01,
)

plt.tight_layout(pad=0.6, h_pad=0.8, w_pad=0.6)

# =============================================================================
# 8. EXPORT
# =============================================================================

out_jpg = OUT_MAIN / f"kde_cars_{ACTIVE_DATASET}_{MODEL_TO_SHOW}.jpg"
fig.savefig(
    out_jpg,
    dpi=250,
    format="jpeg",
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.close(fig)
print(f"✓ Map saved → {out_jpg}")