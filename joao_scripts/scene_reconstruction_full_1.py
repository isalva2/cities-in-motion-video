"""
Scene Reconstruction from Object-Detection Outputs
====================================================
Reproducible example for the UIC Hackathon dataset.

What this script does:
  1. Parses a JSONL file of YOLO detections (YOLOv5n / YOLOv8n / YOLOv10n)
  2. Loops over every *-snapshot.jpg in the same folder as JSONL_PATH
  3. For each image:
       a. Matches it to its nearest detection record
       b. Draws bounding boxes overlaid on the image (4-panel: one per model + combined)
       c. Builds a detection summary table
       d. Generates a scene narrative purely from detection metadata
       e. Saves confidence distribution histograms
     All per-image outputs are prefixed with the image filename stem.
  4. Plots hourly object counts across the full day (once, for the whole JSONL)

Output files (per image, e.g. stem = "1764712201756128484-snapshot"):
  <stem>_detections.csv
  <stem>_fig1_bounding_boxes.png
  <stem>_fig2_detection_table.png
  <stem>_fig4_confidence_dist.png

Output files (once per run, saved next to the images):
  fig3_hourly_counts.png

Requirements:
  pip install pandas matplotlib pillow

Usage:
  1. Set JSONL_PATH below to point to your .jsonl file.
  2. Place all *-snapshot.jpg images in the same folder as the JSONL.
  3. Run:  python scene_reconstruction_full.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# =============================================================================
# 1.  CONFIGURATION  —  only edit ACTIVE_DATASET to switch between datasets
# =============================================================================

BASE_DATA = Path(r"C:/Users/joaob/Dropbox/Documents/hackaton_UIC/hackaton_project/data")
BASE_OUT  = Path(r"C:/Users/joaob/Dropbox/Documents/hackaton_UIC/hackaton_project/data/output")

# Registry of available datasets
# Key         : human-readable label (also used as the output subfolder name)
# jsonl_path  : path to the .jsonl detections file
# image_folder: folder containing *-snapshot.jpg images
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

# ── Change only this line to switch dataset ───────────────────────────────────
ACTIVE_DATASET = "W065_08-17-2025"
# ─────────────────────────────────────────────────────────────────────────────

_cfg         = DATASETS[ACTIVE_DATASET]
JSONL_PATH   = Path(_cfg["jsonl_path"])
IMAGE_FOLDER = Path(_cfg["image_folder"])
IMAGE_PATHS  = sorted(IMAGE_FOLDER.glob("*-snapshot.jpg"))

MODEL_NAMES  = ["YOLOv5n", "YOLOv8n", "YOLOv10n"]
MODEL_COLORS = {"YOLOv5n": "#E85D40", "YOLOv8n": "#3B8BD4", "YOLOv10n": "#3B993B"}

# Output folders — created automatically if they do not exist
# Main: day-level figures (hourly counts, stacked CSV)
# Snapshots: per-image figures and narrative txts
OUT_MAIN      = BASE_OUT / ACTIVE_DATASET
OUT_SNAPSHOTS = OUT_MAIN / "snapshots"
OUT_MAIN.mkdir(parents=True, exist_ok=True)
OUT_SNAPSHOTS.mkdir(parents=True, exist_ok=True)

if not IMAGE_PATHS:
    raise FileNotFoundError(
        f"No *-snapshot.jpg files found in {IMAGE_FOLDER.resolve()}"
    )

print(f"Found {len(IMAGE_PATHS)} image(s) in {IMAGE_FOLDER.resolve()}")
for p in IMAGE_PATHS:
    print(f"  {p.name}")

# =============================================================================
# 2.  HELPERS
# =============================================================================

def parse_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of parsed records."""
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
    return records


def parse_nested_json(value) -> dict | None:
    """The 'value' field is itself a JSON-encoded string — unwrap it."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def get_timestamp_ns(image_path: Path) -> int | None:
    """Extract the leading nanosecond timestamp from filenames like
    1764712201756128484-snapshot.jpg."""
    m = re.match(r"^(\d+)-snapshot", image_path.name)
    return int(m.group(1)) if m else None


LOCAL_TZ = ZoneInfo("America/Chicago")   # handles CST (UTC-6) and CDT (UTC-5) automatically

def ns_to_utc(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)

def ns_to_local(ts_ns: int) -> datetime:
    """Convert nanosecond timestamp to Chicago local time (CST/CDT)."""
    return ns_to_utc(ts_ns).astimezone(LOCAL_TZ)


def build_det_df(models_results: dict) -> pd.DataFrame:
    """Flatten a models_results dict into a per-detection DataFrame."""
    rows = []
    for model in MODEL_NAMES:
        for det in models_results.get(model, {}).get("detections", []):
            bbox = det.get("bbox", [None] * 4)
            rows.append({
                "model"     : model,
                "class"     : det.get("class"),
                "confidence": det.get("confidence"),
                "x_min"     : bbox[0],
                "y_min"     : bbox[1],
                "x_max"     : bbox[2],
                "y_max"     : bbox[3],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["bbox_area"] = (
            (df["x_max"] - df["x_min"]) * (df["y_max"] - df["y_min"])
        )
    return df


def build_narrative(det_df: pd.DataFrame) -> list[str]:
    """Return narrative lines built purely from detection metadata."""
    VEHICLE_CLS = {"car", "truck", "bus", "van", "motorcycle"}
    PERSON_CLS  = {"person", "pedestrian"}
    INFRA_CLS   = {"traffic light", "stop sign", "fire hydrant", "parking meter"}
    BIKE_CLS    = {"bicycle"}

    classes      = sorted(det_df["class"].dropna().unique())
    class_counts = det_df.groupby("class").size()

    found_vehicles = VEHICLE_CLS & set(classes)
    found_persons  = PERSON_CLS  & set(classes)
    found_infra    = INFRA_CLS   & set(classes)
    found_bikes    = BIKE_CLS    & set(classes)

    by_class_model: dict[str, set] = defaultdict(set)
    for _, row in det_df.iterrows():
        by_class_model[row["class"]].add(row["model"])
    consensus_classes = [c for c, ms in by_class_model.items() if len(ms) >= 2]

    lines = []

    if found_vehicles:
        vc = {c: int(class_counts.get(c, 0)) for c in found_vehicles}
        lines.append(
            f"[ROAD SCENE]      Vehicle classes {sorted(found_vehicles)} confirm a road "
            f"or intersection environment. Counts across all models: {vc}."
        )
    if found_persons:
        pc = sum(int(class_counts.get(c, 0)) for c in found_persons)
        lines.append(
            f"[PEDESTRIANS]     {pc} person detection(s) — sidewalk, crosswalk, "
            f"or bus stop area likely adjacent."
        )
    if found_infra:
        lines.append(
            f"[INFRASTRUCTURE]  {sorted(found_infra)} detected — controlled "
            f"intersection or regulated road section."
        )
    if found_bikes:
        lines.append(
            f"[MIXED TRAFFIC]   Bicycles detected — mixed-use road environment."
        )

    high_conf = det_df[det_df["confidence"] >= 0.80]
    low_conf  = det_df[det_df["confidence"] <  0.60]
    lines.append(
        f"[SCENE CLARITY]   {len(high_conf)} high-confidence (≥0.80) detections — "
        f"scene likely well-lit/unobstructed. "
        f"{len(low_conf)} low-confidence detection(s) may be distant/occluded."
    )
    if consensus_classes:
        lines.append(
            f"[CONSENSUS]       ≥2 models agree on: {consensus_classes}. "
            f"Most reliable scene elements."
        )
    if "bbox_area" in det_df.columns and not det_df["bbox_area"].isna().all():
        large = det_df[det_df["bbox_area"] > det_df["bbox_area"].quantile(0.75)]
        if not large.empty:
            lines.append(
                f"[DEPTH]           {len(large)} large bounding box(es) (top 25% by area) "
                f"→ objects close to the camera."
            )
    lines.append(
        "\n[VERDICT]  YES — the scene is reconstructible from detections alone.\n"
        "  The object classes, spatial layout, confidence levels, and cross-model\n"
        "  agreement indicate: urban/suburban intersection, active vehicle traffic,\n"
        "  possible pedestrians, and traffic-control infrastructure."
    )
    return lines


def draw_boxes_on_image(
    ax,
    image_path: Path,
    data: pd.DataFrame,
    title: str,
    single_color: str | None = None,
):
    """Draw one panel: image background + bounding boxes."""
    ax.imshow(Image.open(image_path))
    ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")
    for _, row in data.iterrows():
        col  = single_color or MODEL_COLORS.get(row["model"], "#888888")
        x, y = row["x_min"], row["y_min"]
        w    = row["x_max"] - row["x_min"]
        h    = row["y_max"] - row["y_min"]
        ax.add_patch(patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=col, facecolor=col + "33",
        ))
        ax.text(
            x, max(y - 6, 0),
            f"{row['class']} {row['confidence']:.2f}",
            fontsize=7, color="white",
            bbox=dict(facecolor=col, alpha=0.85, edgecolor="none", pad=2),
        )


# =============================================================================
# 3.  LOAD & PARSE JSONL  (done once — shared by all images)
# =============================================================================

print("\n" + "=" * 65)
print("STEP 1 — Loading JSONL")
print("=" * 65)

raw_records = parse_jsonl(JSONL_PATH)
print(f"  Total records in file : {len(raw_records)}")

detection_rows   = []
model_stats_rows = []

for rec in raw_records:
    if rec.get("name") != "object.detections.all":
        continue
    val = parse_nested_json(rec.get("value"))
    if val is None:
        continue

    img_ts_ns      = val.get("image_timestamp_ns")
    models_results = val.get("models_results", {})

    detection_rows.append({
        "record_timestamp"  : pd.to_datetime(rec.get("timestamp"), utc=True, errors="coerce"),
        "image_timestamp_ns": img_ts_ns,
        "image_datetime"    : ns_to_utc(img_ts_ns) if img_ts_ns else pd.NaT,
        "models_results"    : models_results,
        "meta_vsn"          : rec.get("meta.vsn"),
        "_line"             : rec["_line"],
    })

    for model_name, model_result in models_results.items():
        model_stats_rows.append({
            "image_timestamp_ns"    : img_ts_ns,
            "image_datetime"        : ns_to_utc(img_ts_ns) if img_ts_ns else pd.NaT,
            "model"                 : model_name,
            "total_objects"         : model_result.get("total_objects", 0),
            "inference_time_seconds": model_result.get("inference_time_seconds"),
        })

detections_df  = pd.DataFrame(detection_rows)
model_stats_df = pd.DataFrame(model_stats_rows)

print(f"  Detection events      : {len(detections_df)}")
print(f"  Model×event rows      : {len(model_stats_df)}")

# =============================================================================
# 4.  FIGURE 3 — Hourly counts for the full day  (produced once)
# =============================================================================


print("\n" + "=" * 65)
print("STEP 2 — Hourly object counts (full day, produced once)")
print("=" * 65)
 
model_stats_df["hour"] = (
    pd.to_datetime(model_stats_df["image_datetime"], utc=True)
    .dt.tz_convert("America/Chicago")
    .dt.floor("h")
)
 
hourly = (
    model_stats_df
    .groupby(["hour", "model"])["total_objects"]
    .sum()
    .reset_index()
    .pivot(index="hour", columns="model", values="total_objects")
    .fillna(0)
    .reindex(columns=MODEL_NAMES)
)
# Convert to local hour string first, then sort numerically (0–23).
# Sorting the tz-aware DatetimeIndex sorts by UTC, which puts 19:00 CDT
# (= 00:00 UTC) first on summer datasets — sorting by the local "%H" integer
# ensures the x-axis always runs 00 → 01 → … → 23 regardless of DST offset.
hourly.index = hourly.index.strftime("%H")
hourly = hourly.sort_index(key=lambda idx: idx.astype(int))
 
fig3, ax3 = plt.subplots(figsize=(14, 5), constrained_layout=True)
x      = range(len(hourly))
width  = 0.25
offset = [-width, 0, width]
 
for i, model in enumerate(MODEL_NAMES):
    ax3.bar(
        [xi + offset[i] for xi in x],
        hourly[model],
        width=width,
        color=MODEL_COLORS[model],
        alpha=0.85,
        label=model,
    )
 
ax3.set_xticks(list(x))
ax3.set_xticklabels(hourly.index, rotation=45, ha="right", fontsize=9)
ax3.set_xlabel("Hour (Chicago time)", fontsize=10)
ax3.set_ylabel("Total objects detected", fontsize=10)
ax3.set_title(
    f"Hourly object counts — {JSONL_PATH.stem}",
    fontsize=11, fontweight="bold",
)
ax3.legend(title="Model", fontsize=9)
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top", "right"]].set_visible(False)
 
out_hourly = OUT_MAIN / "fig3_hourly_counts.png"
fig3.savefig(out_hourly, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  → Saved {out_hourly.name}")
 
 



# =============================================================================
# 5.  PER-IMAGE LOOP
# =============================================================================

print("\n" + "=" * 65)
print(f"STEP 3 — Pre-pass: collecting all detections ({len(IMAGE_PATHS)} image(s))")
print("=" * 65)

all_detections: list[pd.DataFrame] = []

for image_path in IMAGE_PATHS:

    img_ts_ns = get_timestamp_ns(image_path)
    if img_ts_ns is None:
        continue

    df_copy = detections_df.copy()
    df_copy["time_diff_s"] = (
        df_copy["image_timestamp_ns"].astype("int64") - img_ts_ns
    ).abs() / 1_000_000_000

    nearest        = df_copy.sort_values("time_diff_s").iloc[0]
    det_df         = build_det_df(nearest["models_results"])

    if det_df.empty:
        continue

    det_df.insert(0, "image_id", image_path.stem)
    all_detections.append(det_df)

# =============================================================================
# 6.  STACKED CSV
# =============================================================================

stacked_csv = OUT_MAIN / "snapshot_detections.csv"
if all_detections:
    stacked_df = pd.concat(all_detections, ignore_index=True)
    stacked_df.to_csv(stacked_csv, index=False)
    print(f"\n  → Saved {stacked_csv.name}  ({len(stacked_df)} rows, {stacked_df['image_id'].nunique()} images)")
else:
    print("\n  ⚠  No detections to write.")
    stacked_df = pd.DataFrame()

# =============================================================================
# 7.  FIGURE 4 — Hourly counts by class group, from matched snapshots only
# =============================================================================

print("\n" + "=" * 65)
print("STEP 4 — Hourly counts by class group (matched snapshots only)")
print("=" * 65)

CLASS_GROUPS = {
    "Cars"           : {"car"},
    "Pedestrians"    : {"person", "pedestrian"},
    "Trucks & Buses" : {"truck", "bus"},
    "Trains"         : {"train"},
}
GROUP_COLORS = {
    "Cars"           : "#3B8BD4",
    "Pedestrians"    : "#E85D40",
    "Trucks & Buses" : "#3B993B",
    "Trains"         : "#9B59B6",
}

if not stacked_df.empty:
    def image_id_to_hour_str(image_id):
        m = re.match(r"^(\d+)-snapshot", image_id)
        if not m:
            return None
        ts_ns = int(m.group(1))
        dt = ns_to_local(ts_ns)
        # Floor to the hour so all snapshots within the same hour share one label
        return dt.replace(minute=0, second=0, microsecond=0).strftime("%H")

    stacked_df["hour_str"] = stacked_df["image_id"].apply(image_id_to_hour_str)

    def assign_group(cls):
        for group, members in CLASS_GROUPS.items():
            if cls in members:
                return group
        return None

    stacked_df["group"] = stacked_df["class"].apply(assign_group)
    fig4_df = stacked_df[stacked_df["group"].notna()].copy()

    hour_labels = sorted(fig4_df["hour_str"].dropna().unique())
    x      = range(len(hour_labels))
    width  = 0.25
    offset = [-width, 0, width]

    fig4, axes4 = plt.subplots(
        4, 1, figsize=(14, 16), constrained_layout=True, sharex=True
    )

    for ax, (group, _) in zip(axes4, CLASS_GROUPS.items()):
        col    = GROUP_COLORS[group]
        grp_df = fig4_df[fig4_df["group"] == group]

        for i, model in enumerate(MODEL_NAMES):
            counts = (
                grp_df[grp_df["model"] == model]
                .groupby("hour_str")
                .size()
                .reindex(hour_labels, fill_value=0)
            )
            ax.bar(
                [xi + offset[i] for xi in x],
                counts.values,
                width=width,
                color=MODEL_COLORS[model],
                alpha=0.85,
                label=model,
            )

        ax.set_title(group, fontsize=11, fontweight="bold", color=col, loc="left")
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(title="Model", fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    # Show one tick every 2 hours
    tick_positions = [i for i, h in enumerate(hour_labels) if int(h) % 2 == 0]

    axes4[-1].set_xticks(tick_positions)
    axes4[-1].set_xticklabels(
        [hour_labels[i] for i in tick_positions],
        rotation=0, ha="center", fontsize=9,
    )
    axes4[-1].set_xlabel("Hour (Chicago time)", fontsize=10)

    fig4.suptitle(
        f"Hourly detections by class group — matched snapshots only\n{JSONL_PATH.stem}",
        fontsize=12, fontweight="bold",
    )
    out_fig4 = OUT_MAIN / "fig4_hourly_by_class.png"
    fig4.savefig(out_fig4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  → Saved {out_fig4.name}")
else:
    print("  ⚠  No stacked detections available — skipping fig4.")

# =============================================================================
# 8.  PER-IMAGE LOOP — narrative + fig1
# =============================================================================

print("\n" + "=" * 65)
print(f"STEP 5 — Per-image figures ({len(IMAGE_PATHS)} image(s))")
print("=" * 65)

for image_path in IMAGE_PATHS:

    stem = image_path.stem
    print(f"\n{'─' * 65}")
    print(f"  Image : {image_path.name}")

    # Match to nearest detection record
    img_ts_ns = get_timestamp_ns(image_path)
    if img_ts_ns is None:
        print(f"  [skip] cannot parse timestamp from filename")
        continue

    img_dt = ns_to_local(img_ts_ns)
    print(f"  Chicago : {img_dt}")

    df_copy = detections_df.copy()
    df_copy["time_diff_s"] = (
        df_copy["image_timestamp_ns"].astype("int64") - img_ts_ns
    ).abs() / 1_000_000_000

    nearest        = df_copy.sort_values("time_diff_s").iloc[0]
    models_results = nearest["models_results"]
    diff_s         = nearest["time_diff_s"]

    print(f"  Nearest record line : {nearest['_line']},  diff : {diff_s:.3f} s")
    if diff_s > 10:
        print("  ⚠  WARNING: nearest record is >10 s away — may not match image")

    # Mark this image's hour on the hourly chart (console only)
    selected_hour = img_dt.strftime("%H:%M")
    if selected_hour in hourly.index:
        print(f"  Hour bucket : {selected_hour} Chicago")

    # Build detection table
    det_df = build_det_df(models_results)
    if det_df.empty:
        print("  [skip] no detections found for this image")
        continue
    det_df.insert(0, "image_id", image_path.stem)

    # ── Scene narrative ───────────────────────────────────────────────────
    narrative_lines = build_narrative(det_df)

    narrative_path = OUT_SNAPSHOTS / f"{stem}_narrative.txt"
    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write(f"Image      : {image_path.name}\n")
        f.write(f"Chicago    : {img_dt}\n")
        f.write(f"Detections : {len(det_df)} rows across {det_df['model'].nunique()} models\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("\n".join(narrative_lines))
        f.write("\n")
    print(f"  → Saved {narrative_path.name}")

    # ── Figure 1 — Bounding boxes overlaid on image ───────────────────────
    fig1, axes1 = plt.subplots(1, 4, figsize=(24, 7), constrained_layout=True)

    for i, model in enumerate(MODEL_NAMES):
        subset = det_df[det_df["model"] == model]
        draw_boxes_on_image(
            axes1[i], image_path, subset,
            f"{model} — {len(subset)} detections",
            MODEL_COLORS[model],
        )

    draw_boxes_on_image(
        axes1[3], image_path, det_df,
        f"All models combined — {len(det_df)} detections",
    )
    axes1[3].legend(
        handles=[
            patches.Patch(
                facecolor=MODEL_COLORS[m] + "55",
                edgecolor=MODEL_COLORS[m],
                label=m,
            )
            for m in MODEL_NAMES
        ],
        loc="upper right", fontsize=8, framealpha=0.85,
    )
    fig1.suptitle(
        f"Scene reconstruction — bounding boxes\n"
        f"{image_path.name}\n"
        f"{img_dt.strftime('%H:%M:%S')} Chicago time",
        fontsize=11, fontweight="bold",
    )
    out1 = OUT_SNAPSHOTS / f"{stem}_fig1_bounding_boxes.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  → Saved {out1.name}")

# =============================================================================
# DONE
# =============================================================================

print("\n" + "=" * 65)
print("All done.")
print(f"  Main outputs       → {OUT_MAIN.resolve()}")
print(f"    fig3_hourly_counts.png")
print(f"    fig4_hourly_by_class.png")
print(f"    snapshot_detections.csv  (all images stacked)")
print(f"  Snapshot outputs   → {OUT_SNAPSHOTS.resolve()}")
print(f"    <stem>_fig1_bounding_boxes.png  (one per image)")
print(f"    <stem>_narrative.txt            (one per image)")
print("=" * 65)