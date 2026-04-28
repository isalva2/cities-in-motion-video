"""
Analysis 3 — Confidence by Hour of Day, one boxplot chart per object class
===========================================================================
Timestamp approach mirrors the working hourly-counts script:
  - Parse image_timestamp_ns into a pandas Series of UTC timestamps
  - Use dt.tz_convert("America/Chicago") + dt.floor("h")
  - Extract local hour as int via strftime("%H") — avoids DST sort issues

Outputs (one PNG per class):
  analysis3_confidence_by_hour_<classname>.png

Requirements:
  pip install pandas matplotlib

Usage:
  Adjust the CONFIGURATION block, then run:
      python analysis3_confidence_by_hour_per_class.py
"""

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DATA = Path(r"C:/Users/joaob/Dropbox/Documents/hackaton_UIC/hackaton_project/data")
BASE_OUT  = Path(r"C:/Users/joaob/Dropbox/Documents/hackaton_UIC/hackaton_project/data/output")

DATASETS = {
    "W06E_12-02-2025": {
        "jsonl_path": BASE_DATA / "W06E_12-02-2025/data/W06E_2025_12_2025-12-02.jsonl",
    },
    "W042_08-09-2025": {
        "jsonl_path": BASE_DATA / "W042_08-09-2025/data/W042_2025_08_2025-08-09.jsonl",
    },
    "W065_08-17-2025": {
        "jsonl_path": BASE_DATA / "W065_08-17-2025/data/W065_2025_08_2025-08-17.jsonl",
    },
}

ACTIVE_DATASET = "W065_08-17-2025"

JSONL_PATH = Path(DATASETS[ACTIVE_DATASET]["jsonl_path"])
OUT_DIR    = BASE_OUT / ACTIVE_DATASET
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = {
    "car"        : "Car",
    "person"     : "Pedestrian",
    "truck"      : "Truck / Bus",
    "bus"        : "Bus",
    "bicycle"    : "Bicycle",
    "motorcycle" : "Motorcycle",
}

# =============================================================================
# LIGHTING PERIOD PALETTE
# =============================================================================

PERIOD_COLORS = {
    "Pre-dawn (0-6)"         : "#4a90d9",
    "Morning twilight (7-8)" : "#e07b39",
    "Daylight (9-14)"        : "#f5c842",
    "Dusk (15-16)"           : "#c0392b",
    "Night (17-23)"          : "#9b59b6",
}
PERIOD_LEGEND = [
    ("Pre-dawn (0-6)",          "#4a90d9"),
    ("Morning twilight (7-8)",  "#e07b39"),
    ("Daylight (9-14)",         "#f5c842"),
    ("Dusk (15-16)",            "#c0392b"),
    ("Night (17-23)",           "#9b59b6"),
]

def hour_color(h: int) -> str:
    if h < 7:  return PERIOD_COLORS["Pre-dawn (0-6)"]
    if h < 9:  return PERIOD_COLORS["Morning twilight (7-8)"]
    if h < 15: return PERIOD_COLORS["Daylight (9-14)"]
    if h < 17: return PERIOD_COLORS["Dusk (15-16)"]
    return            PERIOD_COLORS["Night (17-23)"]

def lighting_period(h: int) -> str:
    if h < 7:  return "Pre-dawn (0-6)"
    if h < 9:  return "Morning twilight (7-8)"
    if h < 15: return "Daylight (9-14)"
    if h < 17: return "Dusk (15-16)"
    return            "Night (17-23)"

# =============================================================================
# LOAD JSONL — collect raw rows with nanosecond timestamps
# =============================================================================

print("=" * 65)
print(f"Loading {JSONL_PATH.name} ...")
print("=" * 65)

rows = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("name") != "object.detections.all":
            continue
        val   = json.loads(r["value"])
        ts_ns = val["image_timestamp_ns"]

        for model, mr in val["models_results"].items():
            for det in mr.get("detections", []):
                cls = det.get("class", "")
                if cls not in TARGET_CLASSES:
                    continue
                rows.append({
                    "ts_ns"     : ts_ns,
                    "cls"       : cls,
                    "confidence": det["confidence"],
                })

df_raw = pd.DataFrame(rows)
print(f"  Raw rows collected : {len(df_raw):,}")
if df_raw.empty:
    raise SystemExit("No matching detections — check TARGET_CLASSES keys.")

# =============================================================================
# TIMESTAMP CONVERSION — mirrors the working hourly-counts approach exactly:
#   1. to_datetime from integer nanoseconds, explicit UTC
#   2. tz_convert to Chicago  (pandas handles CDT/CST automatically)
#   3. floor to hour  (keeps a tz-aware Timestamp per row)
#   4. strftime("%H") → sort by int  (avoids UTC-sort scrambling DST dates)
# =============================================================================

df_raw["ts_utc"]    = pd.to_datetime(df_raw["ts_ns"], unit="ns", utc=True)
df_raw["ts_chi"]    = df_raw["ts_utc"].dt.tz_convert("America/Chicago")
df_raw["hour_utc_str"]  = df_raw["ts_utc"].dt.strftime("%H")        # "00".."23"
df_raw["hour_chi_str"]  = df_raw["ts_chi"].dt.strftime("%H")        # "00".."23"
df_raw["hour"]      = df_raw["hour_str"].astype(int)             # 0..23
df_raw["period"]    = df_raw["hour"].map(lighting_period)

df_raw["hour_utc_str"].unique()
df_raw["hour_chi_str"].unique()
# =============================================================================
# DIAGNOSTICS
# =============================================================================

print()
print("-- Conversion spot-check (5 rows) --------------------------------")
check = df_raw[["ts_ns", "ts_utc", "ts_chi", "hour"]].drop_duplicates(
    subset="hour"
).sort_values("hour").head(5)
for _, row in check.iterrows():
    print(f"  ns={row.ts_ns}  UTC={row.ts_utc}  Chicago={row.ts_chi}  hour={row.hour}")

print()
print("-- Hour distribution (all target-class detections) ---------------")
hc = df_raw["hour"].value_counts().sort_index()
all_hours = list(range(24))
bar_max   = max(hc.values) if len(hc) else 1
for h in all_hours:
    n   = hc.get(h, 0)
    bar = "#" * min(40, int(n / bar_max * 40))
    print(f"  {h:02d}h  {n:>8,}  {bar}")

missing_global = [h for h in all_hours if hc.get(h, 0) == 0]
print()
if missing_global:
    print(f"  WARNING: hours absent across ALL classes: {missing_global}")
    print("  These are genuine data gaps in the JSONL, not a script bug.")
else:
    print("  OK: all 24 hours present in the loaded data.")

print(f"\n  Classes found: {sorted(df_raw['cls'].unique())}")
print()

# =============================================================================
# PLOT HELPER
# =============================================================================

def plot_class(class_key: str, class_label: str, df_cls: pd.DataFrame) -> None:
    hour_data   = [df_cls.loc[df_cls["hour"] == h, "confidence"].values
                   for h in all_hours]
    hour_counts = [len(d) for d in hour_data]

    missing_cls = [h for h, d in zip(all_hours, hour_data) if len(d) == 0]
    if missing_cls:
        print(f"  WARNING [{class_label}]: no detections for hours {missing_cls} "
              f"(genuine data gap)")

    positions_to_plot = [h for h, d in zip(all_hours, hour_data) if len(d) > 0]
    data_to_plot      = [d for d in hour_data if len(d) > 0]
    colors_to_plot    = [hour_color(h) for h in positions_to_plot]

    if not data_to_plot:
        print(f"  [skip] '{class_key}' — no detections at all")
        return

    fig, ax = plt.subplots(figsize=(16, 7), facecolor="white")
    ax.set_facecolor("#f9f9f9")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")

    bp = ax.boxplot(
        data_to_plot,
        positions=positions_to_plot,
        widths=0.65,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2.5, alpha=0.25, linestyle="none"),
        medianprops=dict(color="#111111", linewidth=1.8),
        whiskerprops=dict(linewidth=1.1, color="#555555"),
        capprops=dict(linewidth=1.1, color="#555555"),
        boxprops=dict(linewidth=1.1),
        manage_ticks=False,   # we control ticks — positions map directly to hours
    )
    for patch, flier, col in zip(bp["boxes"], bp["fliers"], colors_to_plot):
        patch.set_facecolor(col)
        patch.set_alpha(0.72)
        flier.set_markerfacecolor(col)
        flier.set_markeredgecolor(col)

    # Hourly mean line
    means   = [d.mean() if len(d) > 0 else np.nan for d in hour_data]
    valid_h = [h for h, m in zip(all_hours, means) if not np.isnan(m)]
    valid_m = [m for m in means if not np.isnan(m)]
    if valid_h:
        ax.plot(valid_h, valid_m, color="#111111", lw=1.6, ls="--",
                marker="D", markersize=4, markerfacecolor="white",
                markeredgecolor="#111111", zorder=6, label="Hourly mean")

    # Period background shading
    for h_start, h_end, col in [
        (0,  6,  "#4a90d9"), (7,  8,  "#e07b39"),
        (9,  14, "#f5c842"), (15, 16, "#c0392b"), (17, 23, "#9b59b6"),
    ]:
        ax.axvspan(h_start - 0.5, h_end + 0.5, color=col, alpha=0.05, zorder=0)

    ax.set_xlim(-0.6, 23.6)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(all_hours)
    ax.set_xticklabels([f"{h:02d}h" for h in all_hours],
                       fontsize=8, rotation=45, ha="right")
    ax.tick_params(colors="#333333", labelsize=8)
    ax.grid(axis="y", alpha=0.20, color="#cccccc")
    ax.grid(axis="x", alpha=0.08, color="#cccccc")

    ax.set_title(
        f"Detection Confidence by Hour of Day  —  Object: {class_label}\n"
        f"{ACTIVE_DATASET}  |  Chicago (CDT/CST)  |  all YOLO models combined  "
        f"|  n = {len(df_cls):,} detections",
        fontsize=12, fontweight="bold", color="#1a1a1a", pad=10,
    )
    ax.set_xlabel("Hour of day (local Chicago time)",
                  fontsize=10, color="#222222", labelpad=8)
    ax.set_ylabel("Confidence", fontsize=10, color="#222222")

    y_bottom = ax.get_ylim()[0]
    for h, n in zip(all_hours, hour_counts):
        label = f"{n:,}" if n >= 1000 else (str(n) if n > 0 else "-")
        ax.text(h, y_bottom - 0.025, label, ha="center", va="top",
                fontsize=6.5, color="#666666", rotation=45)

    period_patches = [mpatches.Patch(facecolor=col, alpha=0.72, label=lbl)
                      for lbl, col in PERIOD_LEGEND]
    mean_handle = plt.Line2D([0], [0], color="#111111", lw=1.6, ls="--",
                             marker="D", markersize=5, markerfacecolor="white",
                             markeredgecolor="#111111", label="Hourly mean")
    ax.legend(handles=period_patches + [mean_handle], loc="lower right",
              fontsize=8, framealpha=0.92, edgecolor="#cccccc",
              title="Lighting period", title_fontsize=8.5)

    # Console summary
    print(f"\n  -- {class_label} ({class_key})  n={len(df_cls):,} --")
    print(f"  {'Hour':>5}  {'n':>7}  {'median':>8}  {'mean':>8}  {'std':>7}")
    print("  " + "-" * 42)
    for h, d in zip(all_hours, hour_data):
        if len(d) == 0:
            print(f"  {h:02d}h    {'no data':>7}")
        else:
            print(f"  {h:02d}h    {len(d):>7,}  {np.median(d):>8.3f}"
                  f"  {d.mean():>8.3f}  {d.std():>7.3f}")

    fig.tight_layout()
    safe_name = class_key.replace("/", "_").replace(" ", "_")
    out = OUT_DIR / f"analysis3_confidence_by_hour_{safe_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Saved {out}")


# =============================================================================
# MAIN LOOP
# =============================================================================

print("=" * 65)
print("Generating per-class hourly boxplot figures ...")
print("=" * 65)

for cls_key, cls_label in TARGET_CLASSES.items():
    df_cls = df_raw[df_raw["cls"] == cls_key].copy()
    if df_cls.empty:
        print(f"  [skip] '{cls_key}' — no detections in this dataset")
        continue
    plot_class(cls_key, cls_label, df_cls)

print("\nDone.")