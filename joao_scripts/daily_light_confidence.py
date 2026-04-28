"""
Analysis 1 — Confidence vs. Bounding Box Area, stratified by lighting period
=============================================================================
Explores whether lighting conditions affect model confidence independently of
object size. The core idea: if two detections have the same bounding box area
(same apparent object size) but different confidence scores, the difference
must come from something else — lighting, weather, contrast.

Lighting periods are derived from the image timestamp (Chicago/CST):
  Pre-dawn          00:00 – 06:59   artificial light only
  Morning twilight  07:00 – 08:59   rapidly changing light
  Daylight          09:00 – 14:59   best visibility window
  Dusk              15:00 – 16:59   low sun angle, glare possible
  Night             17:00 – 23:59   back to artificial light

Outputs:
  analysis1_confidence_vs_bbox_by_period.png

Requirements:
  pip install pandas matplotlib scipy

Usage:
  Set JSONL_PATH and OUT_DIR below, then run:
      python analysis1_confidence_vs_bbox.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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

# ── Change only this line to switch dataset ───────────────────────────────────
ACTIVE_DATASET = "W065_08-17-2025"
# ─────────────────────────────────────────────────────────────────────────────

JSONL_PATH  = Path(DATASETS[ACTIVE_DATASET]["jsonl_path"])
OUT_DIR     = BASE_OUT / ACTIVE_DATASET
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_TZ        = ZoneInfo("America/Chicago")
SAMPLE_PER_PERIOD = 1500    # dots shown in scatter per panel (performance)
ROLLING_WINDOW    = 400     # detections per rolling mean window

# =========================================================I ====================
# LIGHTING PERIOD DEFINITIONS
# =============================================================================

PERIOD_ORDER = [
    "Pre-dawn (0-6)",
    "Morning twilight (7-8)",
    "Daylight (9-14)",
    "Dusk (15-16)",
    "Night (17-23)",
]
PERIOD_COLORS = {
    "Pre-dawn (0-6)"         : "#4a90d9",
    "Morning twilight (7-8)" : "#e07b39",
    "Daylight (9-14)"        : "#f5c842",
    "Dusk (15-16)"           : "#c0392b",
    "Night (17-23)"          : "#9b59b6",
}


def lighting_period(hour: int) -> str:
    if hour < 7:  return "Pre-dawn (0-6)"
    if hour < 9:  return "Morning twilight (7-8)"
    if hour < 15: return "Daylight (9-14)"
    if hour < 17: return "Dusk (15-16)"
    return               "Night (17-23)"


def ns_to_local(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc).astimezone(LOCAL_TZ)


# =============================================================================
# LOAD JSONL
# =============================================================================

print("=" * 65)
print(f"Loading {JSONL_PATH.name} …")
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
        val = json.loads(r["value"])
        dt  = ns_to_local(val["image_timestamp_ns"])

        for model, mr in val["models_results"].items():
            for det in mr.get("detections", []):
                b    = det.get("bbox", [0, 0, 0, 0])
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area <= 0:
                    continue
                rows.append({
                    "hour"      : dt.hour,
                    "period"    : lighting_period(dt.hour),
                    "model"     : model,
                    "cls"       : det["class"],
                    "confidence": det["confidence"],
                    "bbox_area" : area,
                    "log_area"  : np.log10(area),
                })

df = pd.DataFrame(rows)
print(f"  Total detections loaded : {len(df):,}")
print(f"  log₁₀(area) range       : {df['log_area'].min():.2f} – {df['log_area'].max():.2f}")
print()

# =============================================================================
# SAMPLE FOR SCATTER (keep rendering fast)
# =============================================================================

rng          = np.random.default_rng(42)
sample_parts = []
for period in PERIOD_ORDER:
    sub = df[df["period"] == period]
    n   = min(len(sub), SAMPLE_PER_PERIOD)
    if n > 0:
        sample_parts.append(sub.iloc[rng.choice(len(sub), n, replace=False)])
sample = pd.concat(sample_parts, ignore_index=True)

# =============================================================================
# PLOT
# =============================================================================

fig = plt.figure(figsize=(18, 14), facecolor="white")
fig.suptitle(
    "Confidence vs. Bounding Box Area  —  stratified by lighting period\n"
    f"{ACTIVE_DATASET}  |  Chicago (CST)  |  all YOLO models combined",
    fontsize=13, fontweight="bold", color="#1a1a1a", y=0.98,
)

gs     = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.28,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)
axes   = [fig.add_subplot(gs[r, c]) for r, c in [(0,0),(0,1),(0,2),(1,0),(1,1)]]
leg_ax = fig.add_subplot(gs[1, 2])
leg_ax.set_facecolor("white")
leg_ax.axis("off")

# ── Regression summary table ─────────────────────────────────────────────────
print(f"{'Period':<26} {'n':>8} {'mean_conf':>10} {'r':>7} {'slope':>8} {'p':>10}")
print("─" * 73)

for ax, period in zip(axes, PERIOD_ORDER):
    col  = PERIOD_COLORS[period]
    sub  = sample[sample["period"] == period]
    full = df[df["period"] == period]

    if full.empty:
        ax.set_visible(False)
        continue

    # Style
    ax.set_facecolor("#f9f9f9")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")

    # Scatter (sampled)
    ax.scatter(sub["log_area"], sub["confidence"],
               c=col, alpha=0.15, s=7, linewidths=0, rasterized=True)

    # OLS regression line
    slope, intercept, r_val, p_val, _ = stats.linregress(
        full["log_area"], full["confidence"]
    )
    x_fit = np.linspace(full["log_area"].min(), full["log_area"].max(), 200)
    ax.plot(x_fit, slope * x_fit + intercept,
            color="#222222", lw=2.0, ls="--", alpha=0.85, zorder=5)

    # Rolling mean ± std band
    fs = full.sort_values("log_area").copy()
    fs["rm"] = fs["confidence"].rolling(ROLLING_WINDOW, center=True, min_periods=30).mean()
    fs["rs"] = fs["confidence"].rolling(ROLLING_WINDOW, center=True, min_periods=30).std()
    mask = fs["rm"].notna()
    ax.fill_between(
        fs.loc[mask, "log_area"],
        fs.loc[mask, "rm"] - fs.loc[mask, "rs"],
        fs.loc[mask, "rm"] + fs.loc[mask, "rs"],
        color=col, alpha=0.18, zorder=3,
    )
    ax.plot(fs.loc[mask, "log_area"], fs.loc[mask, "rm"],
            color=col, lw=1.6, alpha=0.8, zorder=4)

    # Labels & annotations
    ax.set_title(period, fontsize=10, fontweight="bold", color=col, pad=5)
    ax.set_xlabel("log₁₀(bbox area, px²)", fontsize=8.5, color="black")
    ax.set_ylabel("Confidence",            fontsize=8.5, color="black")
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="black", labelsize=8)
    ax.grid(alpha=0.10, color="#cccccc")

    ax.text(0.04, 0.94, f"r = {r_val:.3f}   slope = {slope:.3f}",
            transform=ax.transAxes, fontsize=8.5, color="#222222")
    ax.text(0.97, 0.05,
            f"n = {len(full):,}\nμ = {full['confidence'].mean():.3f}",
            transform=ax.transAxes, fontsize=8, color="#444444",
            ha="right", va="bottom")

    print(f"{period:<26} {len(full):>8,} {full['confidence'].mean():>10.3f} "
          f"{r_val:>7.3f} {slope:>8.4f} {p_val:>10.2e}")

# ── Reading guide panel ───────────────────────────────────────────────────────
leg_ax.text(0.08, 0.96, "Reading guide",
            fontsize=10, fontweight="bold", color="#1a1a1a",
            transform=leg_ax.transAxes, va="top")

guide_items = [
    ("dots",        f"Individual detections (≤{SAMPLE_PER_PERIOD:,} sampled / period)"),
    ("dashed line", "OLS regression — overall size↔confidence trend"),
    ("solid line",  f"Rolling mean (window = {ROLLING_WINDOW} detections)"),
    ("shaded band", "Rolling mean ± 1 std dev"),
    ("r =",         "Pearson correlation: log area ↔ confidence"),
    ("slope =",     "Confidence gain per log₁₀ area unit"),
    ("μ =",         "Mean confidence for the period"),
]
for i, (lbl, desc) in enumerate(guide_items):
    y = 0.82 - i * 0.11
    leg_ax.text(0.08, y,      lbl,  fontsize=8.5, color="#1a1a1a",
                fontweight="bold", transform=leg_ax.transAxes)
    leg_ax.text(0.08, y-0.05, desc, fontsize=7.8, color="#444444",
                transform=leg_ax.transAxes)

# ── Save ──────────────────────────────────────────────────────────────────────
out = OUT_DIR / "analysis1_confidence_vs_bbox_by_period.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\n  → Saved {out}")