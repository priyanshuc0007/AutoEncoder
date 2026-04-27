"""
draw_architecture.py
====================
Generates a high-quality PNG diagram of the AutoML pipeline architecture.
Output: pipeline_architecture.png  (in the project root)

Run:
    python draw_architecture.py
"""

import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0F1117",   # dark background
    "header":    "#1E2130",   # section header fill
    "pipe":      "#1A3A5C",   # main pipeline step
    "pipe_edge": "#2A7FD4",   # main pipeline border
    "trust":     "#1A3A2A",   # trust pillar step
    "trust_edge":"#2ABD6E",   # trust pillar border
    "io":        "#2D1F3A",   # input / output
    "io_edge":   "#9B5DE5",   # input / output border
    "arrow":     "#6B7DBF",   # connector arrows
    "trust_arr": "#2ABD6E",   # trust side arrows
    "white":     "#E8ECF4",   # primary text
    "dim":       "#8892B0",   # secondary text
    "accent":    "#FFD166",   # highlight text
    "warn":      "#EF476F",   # warning accent
    "good":      "#06D6A0",   # success accent
}

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, fill, edge, radius=0.012, lw=1.8, alpha=1.0, zorder=3):
    """Draw a rounded rectangle."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(rect)
    return rect


def txt(ax, x, y, s, size=8, color=C["white"], ha="center", va="center",
        bold=False, zorder=5, wrap=False):
    weight = "bold" if bold else "normal"
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, zorder=zorder,
            wrap=wrap, family="monospace")


def arrow(ax, x0, y0, x1, y1, color=C["arrow"], lw=1.5,
          style="->", zorder=2):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=zorder,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main drawing
# ─────────────────────────────────────────────────────────────────────────────

FIG_W, FIG_H = 22, 13
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── Title ────────────────────────────────────────────────────────────────────
txt(ax, 0.5, 0.971, "AutoLLM Pipeline Architecture", size=16,
    color=C["white"], bold=True)
txt(ax, 0.5, 0.955, "autollm/  ·  End-to-end text classification with Trust Layer",
    size=9, color=C["dim"])

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — INPUT
# ═══════════════════════════════════════════════════════════════════════════
box(ax, 0.02, 0.875, 0.16, 0.058, C["io"], C["io_edge"])
txt(ax, 0.100, 0.911, ">>  INPUT", size=9.5, bold=True, color=C["accent"])
txt(ax, 0.100, 0.891, "CSV file  ·  label_column  ·  text_column(s)", size=7.5, color=C["dim"])

# ── entry arrow ──
arrow(ax, 0.182, 0.904, 0.210, 0.904, color=C["arrow"], lw=2)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE  (7 steps in a horizontal row then output)
# ═══════════════════════════════════════════════════════════════════════════
STEP_Y   = 0.855  # centre y of step boxes
STEP_H   = 0.100
STEP_W   = 0.088
GAP      = 0.015  # gap between boxes
STEPS_X0 = 0.212  # left edge of first step

steps = [
    ("STEP 1", "Data\nValidation",    "DataValidator\nschema & NaN fixes\ncolumn detection"),
    ("STEP 2", "Data\nIntelligence",  "DataIntelligence\ntask type · imbalance\ntext length · model pick"),
    ("STEP 3", "Data\nPreparation",   "ModelTrainer\ntrain/val split\nLabelEncoder\ntokenizer"),
    ("STEP 4", "Model\nTraining",     "HuggingFace Trainer\n+ Optuna HPO\nprajjwal1/bert-tiny\ndistilbert / bert"),
    ("STEP 5", "Model\nEvaluation",   "ModelEvaluator\nF1 · Accuracy\nLatency\nComposite Score"),
    ("STEP 6", "Best Model\nSelection", "ModelEvaluator\n.compare_models()\n70% F1 + 30% Latency"),
    ("STEP 7", "Report\nGeneration",  "ModelEvaluator\n.generate_report()\nbest_model_report.txt"),
]

step_centres = []
for i, (label, title, detail) in enumerate(steps):
    x = STEPS_X0 + i * (STEP_W + GAP)
    step_centres.append(x + STEP_W / 2)

    box(ax, x, STEP_Y - STEP_H / 2, STEP_W, STEP_H,
        C["pipe"], C["pipe_edge"], radius=0.010)

    # step label (top)
    txt(ax, x + STEP_W / 2, STEP_Y + 0.025, label,
        size=6.5, color=C["accent"], bold=True)
    # main title
    txt(ax, x + STEP_W / 2, STEP_Y + 0.002, title,
        size=8.0, color=C["white"], bold=True)
    # detail (multi-line small)
    for j, line in enumerate(detail.split("\n")):
        txt(ax, x + STEP_W / 2,
            STEP_Y - 0.016 - j * 0.014,
            line, size=6.5, color=C["dim"])

# Arrows between steps
for i in range(len(steps) - 1):
    x_from = STEPS_X0 + i * (STEP_W + GAP) + STEP_W
    x_to   = STEPS_X0 + (i + 1) * (STEP_W + GAP)
    arrow(ax, x_from, STEP_Y, x_to, STEP_Y, lw=2)

# ── exit arrow → OUTPUT ──
last_sx = STEPS_X0 + (len(steps) - 1) * (STEP_W + GAP) + STEP_W
arrow(ax, last_sx, STEP_Y, last_sx + 0.012, STEP_Y, lw=2)

# ── OUTPUT box ──
out_x = last_sx + 0.014
box(ax, out_x, STEP_Y - 0.040, 0.10, 0.080,
    C["io"], C["io_edge"])
txt(ax, out_x + 0.05, STEP_Y + 0.022, "[OK]  OUTPUT", size=8, bold=True, color=C["good"])
for j, line in enumerate([
    "models/<run>/",
    "experiments/<run>/",
    "  best_model_report.txt",
    "  pipeline_state.json",
]):
    txt(ax, out_x + 0.05, STEP_Y - 0.004 - j * 0.014,
        line, size=6.5, color=C["dim"])

# ═══════════════════════════════════════════════════════════════════════════
# TRUST LAYER pillars (below each relevant step)
# ═══════════════════════════════════════════════════════════════════════════
TRUST_Y_TOP  = STEP_Y - STEP_H / 2 - 0.040   # top of trust box
TRUST_BOX_H  = 0.160

trust_pillars = [
    # (step_index,  pillar_no, title, lines)
    (0, "P1", "Data Quality\n(Pillar 1)",
     ["check_data_quality()", "· short texts", "· near-dupes",
      "· label noise", "· tiny classes",
      "→ data_quality_report.txt"]),
    (1, "P2", "Decisions Log\n(Pillar 2)",
     ["DecisionsLogger", "· task detection", "· imbalance choice",
      "· model picked", "· epoch / LR",
      "→ decisions_log.json"]),
    (1, "P3", "Reproducibility\n(Pillar 3)",    # offset right slightly
     ["set_global_seeds(42)", "torch · numpy",
      "transformers · random",
      "(no output file)"]),
    (5, "P4", "Baseline\n(Pillar 4)",
     ["compute_majority_baseline()", "DummyClassifier(most_frequent)",
      "Did model beat random?",
      "→ baseline_comparison.txt"]),
    (6, "P5", "Explainability\n(Pillar 5)",
     ["run_explainability()", "· token importance (attn)",
      "· confidence stats", "· ECE calibration",
      "→ explainability_report.txt"]),
]

# overrides for P3 (share step 1 column but shift slightly)
TRUST_P3_OFFSET = 0.055

for idx, (step_i, pno, title, lines) in enumerate(trust_pillars):
    cx   = step_centres[step_i]
    x_off = TRUST_P3_OFFSET if pno == "P3" else 0
    bx   = cx - STEP_W / 2 + x_off
    bw   = STEP_W

    bot_y = TRUST_Y_TOP - TRUST_BOX_H
    box(ax, bx, bot_y, bw, TRUST_BOX_H,
        C["trust"], C["trust_edge"], radius=0.010)

    # connecting arrow from step bottom to trust box top
    arrow(ax, cx + x_off, STEP_Y - STEP_H / 2,
          cx + x_off, TRUST_Y_TOP,
          color=C["trust_arr"], lw=1.4)

    # pillar number badge
    txt(ax, bx + bw / 2, bot_y + TRUST_BOX_H - 0.015,
        pno, size=7, bold=True, color=C["accent"])

    # title
    for j, ln in enumerate(title.split("\n")):
        txt(ax, bx + bw / 2,
            bot_y + TRUST_BOX_H - 0.030 - j * 0.016,
            ln, size=7.5, bold=(j == 0), color=C["white"])

    # detail lines
    for j, line in enumerate(lines):
        txt(ax, bx + bw / 2,
            bot_y + TRUST_BOX_H - 0.060 - j * 0.014,
            line, size=6.2, color=C["dim"])

# ── Pipeline Tracker label (spans whole pipeline at top of trust zone) ──
tracker_y = TRUST_Y_TOP + 0.008
box(ax, STEPS_X0 - 0.005, tracker_y, 0.760, 0.024,
    "#12192B", C["trust_edge"], radius=0.005, lw=1.2, alpha=0.7)
txt(ax, STEPS_X0 + 0.375, tracker_y + 0.012,
    "P6 · PipelineTracker  —  writes pipeline_state.json after every step  "
    "(status: pending → running → completed | failed)",
    size=7.5, color=C["trust_arr"])

# ═══════════════════════════════════════════════════════════════════════════
# WEB INTERFACES  (right side, below output)
# ═══════════════════════════════════════════════════════════════════════════
wi_x = out_x
wi_y = STEP_Y - STEP_H / 2 - 0.190
box(ax, wi_x, wi_y, 0.106, 0.096,
    C["io"], C["io_edge"], radius=0.010)
txt(ax, wi_x + 0.050, wi_y + 0.072, "INTERFACES", size=7.5, bold=True, color=C["accent"])
for j, line in enumerate([
    "train.py (CLI training)",
    "predict.py (CLI inference)",
    "app_streamlit.py (demo UI)",
    "run_streamlit.bat / .sh",
]):
    txt(ax, wi_x + 0.050, wi_y + 0.055 - j * 0.014,
        line, size=6.5, color=C["dim"])

arrow(ax, out_x + 0.050, STEP_Y - STEP_H / 2, out_x + 0.050, wi_y + 0.088, lw=1.4)

# ═══════════════════════════════════════════════════════════════════════════
# KEY / LEGEND
# ═══════════════════════════════════════════════════════════════════════════
leg_x, leg_y = 0.03, 0.048
box(ax, leg_x - 0.01, leg_y - 0.012, 0.46, 0.085,
    C["header"], "#3A4060", radius=0.008)
txt(ax, leg_x + 0.005, leg_y + 0.060, "LEGEND", size=8, bold=True,
    color=C["white"], ha="left")

legend_items = [
    (C["pipe"],  C["pipe_edge"],  "Main pipeline step  (autollm/*.py)"),
    (C["trust"], C["trust_edge"], "Trust Layer pillar  (autollm/trust/*.py)"),
    (C["io"],    C["io_edge"],    "Input / Output"),
]
for i, (fc, ec, label) in enumerate(legend_items):
    lx = leg_x + 0.005 + i * 0.145
    box(ax, lx, leg_y + 0.018, 0.026, 0.022, fc, ec, radius=0.005)
    txt(ax, lx + 0.030, leg_y + 0.029, label, size=7, color=C["dim"], ha="left")

# ── dataset / model format note ──
txt(ax, 0.55, 0.042,
    "Dataset: CSV  ·  Labels: any string  ·  "
    "Models: HuggingFace AutoModel  ·  "
    "HPO: Optuna  ·  Device: CPU / CUDA",
    size=7, color=C["dim"])

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_file = "pipeline_architecture.png"
plt.tight_layout(pad=0)
plt.savefig(out_file, dpi=180, bbox_inches="tight",
            facecolor=C["bg"], edgecolor="none")
plt.close()
print(f"✅  Saved  →  {out_file}")
