"""
Multi-Dataset YOLO Model Comparison
=====================================
Reads all datasets defined in DATASETS and produces a comprehensive
comparison table across datasets AND models for the metrics requested.

Metrics per (dataset × model):
  1.  Total objects detected
  2.  Cars count
  3.  Bus/trucks count
  4.  Motorcycles count
  5.  Persons count
  6.  Mean bbox area – all objects
  7.  Mean bbox area – cars
  8.  Mean bbox area – bus/trucks
  9.  Mean bbox area – motorcycles
  10. Mean bbox area – persons

Bonus metrics added for model assessment:
  11. Mean confidence – all
  12. Mean confidence – cars
  13. Mean confidence – persons
  14. High-conf detections (≥0.80) share (%)
  15. Low-conf  detections (<0.60) share (%)
  16. Mean inference time (s)
  17. Cross-model consensus rate (%)  — fraction of detections where ≥2 models agree
  18. Unique object classes detected

Outputs:
  <BASE_OUT>/comparison_table.csv
  <BASE_OUT>/comparison_summary.png   (styled figure)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# CONFIGURATION  — copy your paths here
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

MODEL_NAMES = ["YOLOv5n", "YOLOv8n", "YOLOv10n"]

# Class groupings (lower-cased)
CAR_CLS        = {"car"}
BUS_TRUCK_CLS  = {"bus", "truck"}
MOTO_CLS       = {"motorcycle"}
PERSON_CLS     = {"person", "pedestrian"}

OUT_DIR = BASE_OUT / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def parse_jsonl(path: Path) -> list[dict]:
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
                pass
    return records


def parse_nested_json(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def load_detections(jsonl_path: Path) -> pd.DataFrame:
    """Return a flat DataFrame of every detection across all records & models."""
    raw = parse_jsonl(jsonl_path)
    rows = []
    for rec in raw:
        if rec.get("name") != "object.detections.all":
            continue
        val = parse_nested_json(rec.get("value"))
        if val is None:
            continue
        img_ts = val.get("image_timestamp_ns")
        models_results = val.get("models_results", {})

        for model_name in MODEL_NAMES:
            mr = models_results.get(model_name, {})
            inf_time = mr.get("inference_time_seconds")
            for det in mr.get("detections", []):
                bbox = det.get("bbox", [None] * 4)
                cls  = (det.get("class") or "").lower().strip()
                conf = det.get("confidence")
                x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                area = (
                    (x1 - x0) * (y1 - y0)
                    if all(v is not None for v in [x0, y0, x1, y1])
                    else None
                )
                rows.append({
                    "image_ts"      : img_ts,
                    "model"         : model_name,
                    "class"         : cls,
                    "confidence"    : conf,
                    "bbox_area"     : area,
                    "inference_time": inf_time,
                })
    return pd.DataFrame(rows)


def safe_mean(series: pd.Series) -> float:
    s = series.dropna()
    return round(float(s.mean()), 4) if len(s) else float("nan")

def safe_pct(num, denom) -> float:
    return round(100 * num / denom, 2) if denom else float("nan")

def class_match(series: pd.Series, cls_set: set) -> pd.Series:
    return series.isin(cls_set)


def compute_metrics(df: pd.DataFrame, dataset_label: str, model: str) -> dict:
    """Compute all metrics for one (dataset, model) slice."""
    sub = df[df["model"] == model].copy()
    n   = len(sub)

    # --- class masks --------------------------------------------------------
    m_car    = class_match(sub["class"], CAR_CLS)
    m_bt     = class_match(sub["class"], BUS_TRUCK_CLS)
    m_moto   = class_match(sub["class"], MOTO_CLS)
    m_person = class_match(sub["class"], PERSON_CLS)

    # --- cross-model consensus (fraction of frames where ≥2 models see same class) -
    # We compute per-frame: for each class, how many models found it?
    consensus_rate = float("nan")
    if not df.empty and "image_ts" in df.columns:
        try:
            frames = df.groupby(["image_ts", "class"])["model"].nunique()
            total_class_frames = len(frames)
            consensus = int((frames >= 2).sum())
            consensus_rate = safe_pct(consensus, total_class_frames)
        except Exception:
            pass

    # --- unique classes -------------------------------------------------------
    unique_cls = sorted(sub["class"].dropna().unique().tolist())

    # --- inference time -------------------------------------------------------
    mean_inf = safe_mean(sub["inference_time"].drop_duplicates())  # one per frame

    return {
        "Dataset"              : dataset_label,
        "Model"                : model,
        # counts
        "N_objects"            : n,
        "N_cars"               : int(m_car.sum()),
        "N_bus_truck"          : int(m_bt.sum()),
        "N_motorcycle"         : int(m_moto.sum()),
        "N_persons"            : int(m_person.sum()),
        # bbox areas
        "BBox_all_mean"        : safe_mean(sub["bbox_area"]),
        "BBox_car_mean"        : safe_mean(sub.loc[m_car, "bbox_area"]),
        "BBox_bus_truck_mean"  : safe_mean(sub.loc[m_bt,  "bbox_area"]),
        "BBox_moto_mean"       : safe_mean(sub.loc[m_moto,"bbox_area"]),
        "BBox_person_mean"     : safe_mean(sub.loc[m_person,"bbox_area"]),
        # confidence
        "Conf_all_mean"        : safe_mean(sub["confidence"]),
        "Conf_car_mean"        : safe_mean(sub.loc[m_car,    "confidence"]),
        "Conf_person_mean"     : safe_mean(sub.loc[m_person, "confidence"]),
        "High_conf_pct"        : safe_pct((sub["confidence"] >= 0.80).sum(), n),
        "Low_conf_pct"         : safe_pct((sub["confidence"] <  0.60).sum(), n),
        # timing & diversity
        "Mean_inf_time_s"      : mean_inf,
        "Consensus_rate_pct"   : consensus_rate,
        "Unique_classes"       : len(unique_cls),
        "Class_list"           : ", ".join(unique_cls),
    }


# =============================================================================
# MAIN — iterate all datasets & models
# =============================================================================

print("=" * 65)
print("YOLO Multi-Dataset Comparison")
print("=" * 65)

all_metrics = []

for ds_label, ds_cfg in DATASETS.items():
    jsonl_path = Path(ds_cfg["jsonl_path"])
    print(f"\n▶  Loading {ds_label} …")
    if not jsonl_path.exists():
        print(f"   [SKIP] File not found: {jsonl_path}")
        continue
    df = load_detections(jsonl_path)
    if df.empty:
        print("   [SKIP] No detection records found.")
        continue
    print(f"   {len(df):,} detections loaded.")
    for model in MODEL_NAMES:
        m = compute_metrics(df, ds_label, model)
        all_metrics.append(m)
        print(f"   {model}: {m['N_objects']:,} objects | "
              f"{m['N_cars']} cars | {m['N_persons']} persons | "
              f"conf {m['Conf_all_mean']:.3f}")

if not all_metrics:
    print("\n[ERROR] No metrics computed — check your paths.")
    raise SystemExit(1)

results_df = pd.DataFrame(all_metrics)

# Save CSV
csv_path = OUT_DIR / "comparison_table.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n✔  CSV saved → {csv_path}")

# =============================================================================
# FIGURE — styled comparison table
# =============================================================================

# Columns to display in the figure (exclude verbose Class_list)
DISPLAY_COLS = [
    "Dataset", "Model",
    "N_objects", "N_cars", "N_bus_truck", "N_motorcycle", "N_persons",
    "BBox_all_mean", "BBox_car_mean", "BBox_bus_truck_mean",
    "BBox_moto_mean", "BBox_person_mean",
    "Conf_all_mean", "Conf_car_mean", "Conf_person_mean",
    "High_conf_pct", "Low_conf_pct",
    "Mean_inf_time_s", "Consensus_rate_pct", "Unique_classes",
]

HEADERS = [
    "Dataset", "Model",
    "# Objects", "# Cars", "# Bus/Truck", "# Moto", "# Persons",
    "BBox All\n(mean px²)", "BBox Car\n(mean px²)", "BBox Bus/Trk\n(mean px²)",
    "BBox Moto\n(mean px²)", "BBox Person\n(mean px²)",
    "Conf All\n(mean)", "Conf Car\n(mean)", "Conf Person\n(mean)",
    "High Conf\n≥0.80 (%)", "Low Conf\n<0.60 (%)",
    "Inf Time\n(s)", "Consensus\n(%)", "Unique\nClasses",
]

disp = results_df[DISPLAY_COLS].copy()

# Format floats nicely
float_cols = [c for c in DISPLAY_COLS if disp[c].dtype == float]
for c in float_cols:
    disp[c] = disp[c].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "—")

cell_text = disp.values.tolist()

# ── Color scheme ──────────────────────────────────────────────────────────────
MODEL_PALETTE = {
    "YOLOv5n" : "#E85D40",
    "YOLOv8n" : "#3B8BD4",
    "YOLOv10n": "#2EA84F",
}
DS_PALETTE = {
    list(DATASETS.keys())[0]: "#F0F4FF",
    list(DATASETS.keys())[1]: "#FFF8F0",
    list(DATASETS.keys())[2]: "#F0FFF4",
}

row_colors = []
for row in disp.itertuples(index=False):
    ds_bg = DS_PALETTE.get(row.Dataset, "#FFFFFF")
    row_colors.append([ds_bg] * len(DISPLAY_COLS))

# Column header colors
col_colors = ["#1E2A3A"] * len(DISPLAY_COLS)

# ── Build figure ──────────────────────────────────────────────────────────────
SCALE_FACTOR = 0.65   # <- increase to make all figures larger (e.g. 1.0 = base size)

n_rows = len(disp)
n_cols = len(DISPLAY_COLS)

fig_w = max(28, n_cols * 1.55) * SCALE_FACTOR
fig_h = (max(6, n_rows * 0.55 + 3.5)) * SCALE_FACTOR

fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
ax  = fig.add_subplot(111)
ax.set_facecolor("white")
ax.axis("off")

# Title
fig.text(
    0.5, 0.97,
    "YOLO Model Comparison — Multi-Dataset",
    ha="center", va="top",
    fontsize=18, fontweight="bold", color="black",
    fontfamily="monospace",
)
fig.text(
    0.5, 0.93,
    "Object counts · BBox areas · Confidence · Inference speed · Cross-model consensus",
    ha="center", va="top",
    fontsize=10, color="#444444",
    fontfamily="monospace",
)

tbl = ax.table(
    cellText  = cell_text,
    colLabels = HEADERS,
    cellLoc   = "center",
    loc       = "center",
    cellColours = row_colors,
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.55)

# Style header — light grey background, black bold text
for j in range(n_cols):
    cell = tbl[0, j]
    cell.set_facecolor("#DDDDDD")
    cell.set_text_props(color="black", fontweight="bold", fontsize=7)
    cell.set_edgecolor("#AAAAAA")

# Style data rows — colour the Model cell with a soft tint, black text elsewhere
for i in range(1, n_rows + 1):
    row_model = cell_text[i - 1][1]  # "Model" column index = 1
    model_col = MODEL_PALETTE.get(row_model, "#888888")
    for j in range(n_cols):
        cell = tbl[i, j]
        cell.set_edgecolor("#BBBBBB")
        cell.set_text_props(color="black")
        if j == 1:  # Model column
            cell.set_facecolor(model_col + "30")
            cell.set_text_props(color=model_col, fontweight="bold")

# Legend for models
legend_x = 0.01
for model, col in MODEL_PALETTE.items():
    fig.text(legend_x, 0.02, f"■ {model}", color=col,
             fontsize=9, fontfamily="monospace", fontweight="bold")
    legend_x += 0.10

png_path = OUT_DIR / "comparison_summary.png"
plt.savefig(png_path, dpi=160, bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"✔  Figure saved → {png_path}")

# =============================================================================
# ALSO: per-metric bar charts (one figure per key metric group)
# =============================================================================

METRIC_GROUPS = {
    "Object Counts": ["N_objects", "N_cars", "N_bus_truck", "N_motorcycle", "N_persons"],
    "Mean BBox Area (px2)": ["BBox_all_mean", "BBox_car_mean", "BBox_bus_truck_mean", "BBox_moto_mean", "BBox_person_mean"],
    "Confidence": ["Conf_all_mean", "Conf_car_mean", "Conf_person_mean", "High_conf_pct", "Low_conf_pct"],
    "Performance & Diversity": ["Mean_inf_time_s", "Consensus_rate_pct", "Unique_classes"],
}

GROUP_YLABELS = {
    "Object Counts": "Count (x 1,000)",
    "Mean BBox Area (px2)": "Mean Area (x 1,000 px2)",
    "Confidence": "Value",
    "Performance & Diversity": "Value",
}

# Groups where raw values are divided by 1000 before plotting
SCALE_Y_1000 = {"Object Counts", "Mean BBox Area (px2)"}

# Human-readable titles for each metric (used as subplot titles)
METRIC_TITLES = {
    "N_objects"            : "# Objects",
    "N_cars"               : "# Cars",
    "N_bus_truck"          : "# Bus/Truck",
    "N_motorcycle"         : "# Motorcycle",
    "N_persons"            : "# Persons",
    "BBox_all_mean"        : "BBox mean (all)",
    "BBox_car_mean"        : "BBox mean (car)",
    "BBox_bus_truck_mean"  : "BBox mean (bus/truck)",
    "BBox_moto_mean"       : "BBox mean (motorcycle)",
    "BBox_person_mean"     : "BBox mean (person)",
    "Conf_all_mean"        : "CI mean (all)",
    "Conf_car_mean"        : "CI mean (car)",
    "Conf_person_mean"     : "CI mean (person)",
    "High_conf_pct"        : "High CI >= 0.80 (%)",
    "Low_conf_pct"         : "Low CI < 0.60 (%)",
    "Mean_inf_time_s"      : "Inference time (s)",
    "Consensus_rate_pct"   : "Consensus rate (%)",
    "Unique_classes"       : "Unique classes",
}

# Groups that should be laid out in 2 rows instead of 1
TWO_ROW_GROUPS = {"Confidence", "Object Counts", "Mean BBox Area (px2)"}

numeric_df = results_df.copy()
for c in float_cols:
    numeric_df[c] = pd.to_numeric(results_df[c], errors="coerce")

datasets_list = list(DATASETS.keys())
n_ds   = len(datasets_list)

for group_name, metrics in METRIC_GROUPS.items():
    n_metrics = len(metrics)

    if group_name in TWO_ROW_GROUPS:
        n_cols_layout = int(np.ceil(n_metrics / 2))
        n_rows_layout = 2
    else:
        n_cols_layout = n_metrics
        n_rows_layout = 1

    fig2, axes = plt.subplots(
        n_rows_layout, n_cols_layout,
        figsize=(4.5 * n_cols_layout * SCALE_FACTOR, 5 * n_rows_layout * SCALE_FACTOR),
        facecolor="white",
        squeeze=False,
    )

    # Hide any unused axes (when n_metrics < n_rows*n_cols)
    axes_flat = axes.flatten()
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig2.suptitle(group_name, color="black", fontsize=14,
                  fontweight="bold", fontfamily="monospace", y=1.02)

    x = np.arange(n_ds)
    bar_w = 0.22

    for idx, metric in enumerate(metrics):
        ax2 = axes_flat[idx]
        ax2.set_facecolor("white")
        ax2.spines[:].set_color("#AAAAAA")
        ax2.tick_params(colors="black", labelsize=7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [d.replace("_", "\n") for d in datasets_list],
            fontsize=7, color="black",
        )
        ax2.set_title(
            METRIC_TITLES.get(metric, metric.replace("_", " ")),
            color="black", fontsize=8, pad=6,
        )
        ax2.set_ylabel(GROUP_YLABELS[group_name], color="black", fontsize=7)
        ax2.yaxis.label.set_color("black")
        ax2.grid(axis="y", color="#DDDDDD", linewidth=0.5, linestyle="--")

        for mi, model in enumerate(MODEL_NAMES):
            vals = []
            for ds in datasets_list:
                row = numeric_df[
                    (numeric_df["Dataset"] == ds) & (numeric_df["Model"] == model)
                ]
                v = float(row[metric].iloc[0]) if not row.empty and pd.notna(row[metric].iloc[0]) else 0.0
                if group_name in SCALE_Y_1000:
                    v = v / 1000.0
                vals.append(v)
            offset = (mi - 1) * bar_w
            bars = ax2.bar(
                x + offset, vals, bar_w,
                color=MODEL_PALETTE[model],
                alpha=0.85, label=model,
                zorder=3,
            )
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{v:,.2f}",
                        ha="center", va="bottom",
                        fontsize=5.5, color="black",
                    )

        # Legend only on the first subplot, bottom right
        if idx == 0:
            ax2.legend(
                MODEL_NAMES,
                fontsize=6, loc="lower right",
                facecolor="white", edgecolor="#AAAAAA",
                labelcolor="black",
            )

    plt.tight_layout()
    safe_name = group_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "").replace("2", "2")
    bar_path = OUT_DIR / f"bars_{safe_name}.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Bar chart saved -> {bar_path}")

print("\n✅  All done. Outputs in:", OUT_DIR.resolve())