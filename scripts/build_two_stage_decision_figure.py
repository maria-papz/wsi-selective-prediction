from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build_figures"))
from build_m1_study_design_figure import box, arrow, swimlane, DARK_PURPLE, BRIGHT_PURPLE, MID_PURPLE, PALE_PURPLE

BASE = Path("/cs/student/project_msc/2025/aibh/mpapageo")
OUT = BASE / "outputs/m3_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

TRUNK_CX = 3.2
RECT_W, RECT_H = 3.4, 0.85
DIAMOND_W, DIAMOND_H = 3.2, 1.0
OUTCOME_W, OUTCOME_H = 4.3, 1.05
SIDE_CX = 8.35
GAP = 0.55  # arrow length between consecutive trunk nodes' facing edges

QUESTION_FC, QUESTION_EC = PALE_PURPLE, BRIGHT_PURPLE
GROUNDTRUTH_FC, GROUNDTRUTH_EC = "#e9eaf2", "#5c5c72"
STATE_FC, STATE_EC = "#f3ecfb", DARK_PURPLE
EXCLUDED_FC, EXCLUDED_EC = "white", DARK_PURPLE
COUNTED_FC, COUNTED_EC = PALE_PURPLE, "#7a3fa6"


def diamond(ax, cx, cy, w, h, text, fc=QUESTION_FC, ec=QUESTION_EC, fontsize=10.5):
    pts = [(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)]
    ax.add_patch(mpatches.Polygon([(x + 0.05, y - 0.05) for x, y in pts], closed=True,
                                    fc="#000000", ec="none", alpha=0.10, zorder=2.5))
    ax.add_patch(mpatches.Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.0, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=4)


def branch_tag(ax, x, y, text, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=9.5, color=DARK_PURPLE,
             fontweight="bold", style="italic", zorder=4)


def legend_entry(ax, x, y, shape, fc, ec, dashed, text):
    cx, cy = x + 0.25, y
    if shape == "diamond":
        w, h = 0.5, 0.36
        pts = [(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)]
        ax.add_patch(mpatches.Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.2, zorder=3))
    else:
        w, h = 0.5, 0.34
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
            fc=fc, ec=ec, lw=1.2, linestyle="dashed" if dashed else "solid", zorder=3))
    ax.text(x + 0.62, y, text, ha="left", va="center", fontsize=9.7, zorder=3)


def stage_band(ax, x_left, x_right, y_bottom, y_top, label, color):
    swimlane(ax, x_left, x_right, y_bottom, y_top, color)
    ax.text(x_left + 0.15, y_top - 0.12, label, ha="left", va="top",
             fontsize=13, color=DARK_PURPLE, fontweight="bold", zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(13.0, 14.5))
    ax.set_xlim(-0.3, 10.8)
    ax.axis("off")
    cursor = [13.6]

    def drop(h_next):
        top = cursor[0]
        cy = top - GAP - h_next / 2
        arrow(ax, (TRUNK_CX, top), (TRUNK_CX, cy + h_next / 2))
        cursor[0] = cy - h_next / 2
        return cy

    def rect(cy, text, w=RECT_W, h=RECT_H, **kw):
        box(ax, TRUNK_CX - w / 2, cy - h / 2, w, h, text, fontsize=10.5, **kw)

    def decision(cy, text, yes_side, side_text, side_style, ground_truth=False):
        if ground_truth:
            diamond(ax, TRUNK_CX, cy, DIAMOND_W, DIAMOND_H, text, fc=GROUNDTRUTH_FC, ec=GROUNDTRUTH_EC)
        else:
            diamond(ax, TRUNK_CX, cy, DIAMOND_W, DIAMOND_H, text)
        fc, ec, dashed = side_style
        right_label = "Yes" if yes_side == "right" else "No"
        down_label = "No" if yes_side == "right" else "Yes"
        arrow(ax, (TRUNK_CX + DIAMOND_W / 2, cy), (SIDE_CX - OUTCOME_W / 2, cy))
        branch_tag(ax, TRUNK_CX + DIAMOND_W / 2 + 0.4, cy + 0.22, right_label)
        branch_tag(ax, TRUNK_CX + 0.3, cy - DIAMOND_H / 2 - 0.28, down_label)
        box(ax, SIDE_CX - OUTCOME_W / 2, cy - OUTCOME_H / 2, OUTCOME_W, OUTCOME_H, side_text,
            fontsize=10.5, fc=fc, ec=ec, linestyle="dashed" if dashed else "solid",
            shadow=not dashed)

    band_x_left, band_x_right = -0.3, 10.8

    # =========================================================================
    # Stage 1: IDH classification
    # =========================================================================
    stage1_top = cursor[0] + 0.5

    y = cursor[0] - RECT_H / 2
    rect(y, "Full evaluation population", fc=MID_PURPLE, fontweight="bold")
    cursor[0] = y - RECT_H / 2

    # --- Decision 1: Stage-1 deferral ---
    y = drop(DIAMOND_H)
    decision(y, "Predictive entropy $>\\,c_1$?",
             yes_side="right",
             side_text="Deferred\nExcluded from automated accuracy",
             side_style=(EXCLUDED_FC, EXCLUDED_EC, True))
    y = drop(RECT_H)
    rect(y, "Retained", fc=STATE_FC, ec=STATE_EC)

    # --- Decision 2: Stage-1 call vs. true label ---
    y = drop(DIAMOND_H)
    decision(y, "Predicted mutant?",
             yes_side="down",
             side_text="Predicted wildtype\nExit -- scored against true IDH label",
             side_style=(COUNTED_FC, COUNTED_EC, False))
    y = drop(RECT_H)
    rect(y, "Predicted mutant", fc=STATE_FC, ec=STATE_EC)

    # --- Decision 3: predicted-mutant vs. true label ---
    y = drop(DIAMOND_H)
    decision(y, "True label: mutant?",
             yes_side="down",
             side_text="False positive (actually wildtype)\nCounted incorrect regardless of Stage 2",
             side_style=(COUNTED_FC, COUNTED_EC, False), ground_truth=True)
    y = drop(RECT_H)
    rect(y, "True positive -- proceeds to Stage 2", fc=STATE_FC, ec=STATE_EC)

    stage1_bottom = cursor[0] - 0.1
    stage_band(ax, band_x_left, band_x_right, stage1_bottom, stage1_top,
               "STAGE 1", "#f8f4fc")

    # =========================================================================
    # Stage 2: 1p/19q codeletion classification
    # =========================================================================
    stage2_top = stage1_bottom

    # --- Decision 4: codeletion label availability ---
    y = drop(DIAMOND_H)
    decision(y, "Codeletion label available?",
             yes_side="down",
             side_text="No codeletion label available\nExcluded from Stage-2 accuracy",
             side_style=(EXCLUDED_FC, EXCLUDED_EC, True), ground_truth=True)
    y = drop(RECT_H)
    rect(y, "Evaluation population", fc=STATE_FC, ec=STATE_EC)

    # --- Decision 5: Stage-2 deferral ---
    y = drop(DIAMOND_H)
    decision(y, "Predictive entropy $>\\,c_2$?",
             yes_side="right",
             side_text="Deferred\nExcluded from automated accuracy",
             side_style=(EXCLUDED_FC, EXCLUDED_EC, True))
    y = drop(RECT_H)
    rect(y, "Retained\nScored against codeletion label",
         fc=MID_PURPLE, ec=BRIGHT_PURPLE, fontweight="bold")

    stage2_bottom = cursor[0] - 0.35
    stage_band(ax, band_x_left, band_x_right, stage2_bottom, stage2_top,
               "STAGE 2", "white")

    # Legend spells out what every shape/colour/border means
    legend_top = stage2_bottom - 0.55
    ax.text(0.0, legend_top, "KEY", ha="left", va="top", fontsize=11,
             color=DARK_PURPLE, fontweight="bold")
    row1_y = legend_top - 0.5
    row2_y = row1_y - 0.55
    legend_entry(ax, 0.0, row1_y, "diamond", QUESTION_FC, BRIGHT_PURPLE, False,
                 "Pipeline decision (its own model output)")
    legend_entry(ax, 5.6, row1_y, "diamond", GROUNDTRUTH_FC, GROUNDTRUTH_EC, False,
                 "Evaluation-only check (needs ground truth)")
    legend_entry(ax, 0.0, row2_y, "box", EXCLUDED_FC, EXCLUDED_EC, True,
                 "Excluded from automated accuracy")
    legend_entry(ax, 5.6, row2_y, "box", COUNTED_FC, COUNTED_EC, False,
                 "Counted toward automated accuracy")
    legend_bottom = row2_y - 0.35

    ax.set_ylim(legend_bottom - 0.1, stage1_top + 0.15)

    for ext in ("pdf", "png"):
        out_path = OUT / f"two_stage_decision_flow.{ext}"
        kwargs = {"dpi": 300} if ext == "png" else {}
        plt.savefig(out_path, **kwargs)
        print(f"Saved {out_path}")

    originals = [t.get_text() for t in ax.texts]
    for t in ax.texts:
        t.set_text(t.get_text().replace("&", r"\&"))
    fig.savefig(OUT / "two_stage_decision_flow.pgf", backend="pgf")
    for t, orig in zip(ax.texts, originals):
        t.set_text(orig)
    plt.close()


if __name__ == "__main__":
    main()
