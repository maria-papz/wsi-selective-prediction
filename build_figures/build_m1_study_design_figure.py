# scripts/build_m1_study_design_figure.py
"""
Methods Chapter 3.1 study-design overview figure. Same visual language as
the Results chapter's Figure 1 (build_r0_figures.py): UCL purple palette,
rounded FancyBboxPatch boxes, serif font, pgf export for \\input{} into the
thesis. Cohort counts recomputed live from the same loaders as Figure 1 --
nothing here is hardcoded, so the two figures can never silently drift
apart.

Depicts the actual study design as a fan-out/fan-in pipeline -- NOT a
straight-line sequence and NOT a visual table of contents: six uncertainty
signals are genuinely parallel (not sequential steps), selective prediction
and conformal prediction are two independent decision mechanisms, and
transport/fairness are two independent testing axes applied to both. No
chapter section numbers are embedded in the figure itself -- that belongs
in the LaTeX caption, not baked into the image.

Input:
    data/raw/ebrains/clinical/annotation.csv
    data/raw/tcga/clinical/{tcga_demographics.tsv, tcga_slides_clean.csv,
                            tcga_idh_final.csv, tcga_manifest.csv}
    data/raw/ipd_brain/clinical/ipd_brain_v1.csv
    outputs/ipd_brain_curation/patient_level_curation.csv
    data/assets/wsi_example_ipd_brain_0010d.png (user-supplied WSI thumbnail)

Output:
    outputs/m1_figures/figure_study_design_overview.png
    outputs/m1_figures/figure_study_design_overview.pdf
    outputs/m1_figures/figure_study_design_overview.pgf
"""
import warnings
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_r0_figures import load_ebrains, load_tcga, load_ipd_brain  # noqa: E402

OUT = Path("outputs/m1_figures")
OUT.mkdir(parents=True, exist_ok=True)
WSI_THUMB = Path("data/assets/wsi_example_ipd_brain_0010d.png")

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

# UCL brand palette -- identical to build_r0_figures.py
DARK_PURPLE = "#361a54"
BRIGHT_PURPLE = "#993bff"
MID_PURPLE = "#ba82ff"
LIGHT_PURPLE = "#ddbdff"
PALE_PURPLE = "#eedeff"
GREY = "#4a4a4a"

# Connector colour -- a muted purple-grey instead of flat engineering-diagram
# black/grey, so the lines read as part of the same palette as the boxes
# rather than a separate schematic layer drawn on top of them.
LINE_COLOR = "#7a6291"


def box(ax, x, y, w, h, text, fc=None, ec=None, fontsize=10, fontweight="normal",
        linestyle="solid", shadow=True):
    fc = PALE_PURPLE if fc is None else fc
    ec = DARK_PURPLE if ec is None else ec
    if shadow:
        # Subtle drop shadow -- a soft offset duplicate, not a hard outline --
        # gives the boxes real depth instead of sitting flat on the page,
        # the single biggest cue that separates a modern card/tile layout
        # from a bare 1980s-style box-and-line schematic.
        sh = mpatches.FancyBboxPatch(
            (x + 0.045, y - 0.045), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.11",
            fc="#000000", ec="none", alpha=0.10, zorder=2.5,
        )
        ax.add_patch(sh)
    r = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.11",
        fc=fc, ec=ec, lw=1.0, zorder=3, linestyle=linestyle,
    )
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, zorder=4, linespacing=1.3)
    return x, y, w, h


def arrow(ax, xy_from, xy_to, color=None, rad=0.0, style="-|>", ls="solid"):
    color = LINE_COLOR if color is None else color
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5,
                                 linestyle=ls, mutation_scale=16, shrinkA=0, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"),
                zorder=2)


def elbow_arrow(ax, x1, y1, x2, y2, y_bend, color=None, lw=1.5):
    color = LINE_COLOR if color is None else color
    ax.plot([x1, x1, x2], [y1, y_bend, y_bend], color=color, lw=lw, zorder=2,
             solid_joinstyle="round", solid_capstyle="round")
    ax.annotate("", xy=(x2, y2), xytext=(x2, y_bend),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                 mutation_scale=16, shrinkA=0, shrinkB=2),
                zorder=2)


def swimlane(ax, x_left, x_right, y_bottom, y_top, color="#f6f2fc"):
    ax.add_patch(mpatches.Rectangle((x_left, y_bottom), x_right - x_left, y_top - y_bottom,
                                     fc=color, ec="none", zorder=0))


def main():
    _, ef = load_ebrains()
    _, _, _, _, tf = load_tcga()
    _, ipf = load_ipd_brain()

    BOX_FONTSIZE = 11.0


    ROW1_W = 2.3            # uniform width for the other three row-1 boxes
    PREPROC_W = 2.9
    XLIM_RIGHT = 10.5 + (PREPROC_W - ROW1_W)


    XLIM_LEFT_PROVISIONAL = -3.0
    fig, ax = plt.subplots(figsize=(XLIM_RIGHT - XLIM_LEFT_PROVISIONAL, 10.75))
    ax.set_xlim(XLIM_LEFT_PROVISIONAL, XLIM_RIGHT)
    ax.set_ylim(1.65, 12.4)
    ax.axis("off")


    GAP = 0.3
    ROW_GAP = 0.55


    fig.canvas.draw()
    RENDERER = fig.canvas.get_renderer()
    ROW_LABELS = ["MODEL TRAINING", "EXTERNAL VALIDATION", "UNCERTAINTY ESTIMATION",
                  "DECISION MECHANISMS", "ROBUSTNESS TESTING"]
    TAG_FONTSIZE, TAG_H, TAG_PAD = BOX_FONTSIZE, 0.4, 0.14

    def text_width(text, fontsize):
        probe = ax.text(0, 0, text, fontsize=fontsize, fontweight="bold")
        bbox = probe.get_window_extent(renderer=RENDERER)
        inv = ax.transData.inverted()
        w = inv.transform((bbox.x1, 0))[0] - inv.transform((bbox.x0, 0))[0]
        probe.remove()
        return w

    widest_label = max(text_width(t, TAG_FONTSIZE) for t in ROW_LABELS)
    TAG_X = -(widest_label + 2 * TAG_PAD + 0.2)  # +0.2 clearance before x=0
    XLIM_LEFT = TAG_X - 0.15
    ax.set_xlim(XLIM_LEFT, XLIM_RIGHT)
    # Re-sync figsize to the now-final xlim range -- the provisional width
    # above was just a placeholder to get a renderer; keeping the two matched
    # is what keeps every fixed-point fontsize elsewhere sized correctly
    # relative to its box under the established 1-inch-per-data-unit rule.
    fig.set_size_inches(XLIM_RIGHT - XLIM_LEFT, 10.75)
    # ...which also means the renderer above is now stale (it reflects the
    # old canvas size) -- every row_label() call below measures its own tag
    # against a fresh one instead of reusing it.
    fig.canvas.draw()
    RENDERER = fig.canvas.get_renderer()

    def row_band(y, h, color="#f6f2fc"):
        # Alternating full-width background band, sized from this row's own
        # (y, h) plus the shared ROW_GAP so it always meets its neighbours
        # exactly at the row-to-row midpoint -- no band ever needs its own
        # hand-picked extent.
        swimlane(ax, XLIM_LEFT, XLIM_RIGHT, y - ROW_GAP / 2, y + h + ROW_GAP / 2, color)

    def row_label(y, h, text):
        # A tag in the dedicated left margin, vertically centred on this
        # row's own midline -- never in the row-to-row gap the connectors
        # use, so it cannot collide with one regardless of that row's own
        # connector layout. Light fill with dark text (not white-on-solid,
        # which read as a harsh warning-label contrast rather than a section
        # tag) -- soft enough to sit beside the pale band without competing
        # with the actual content boxes for attention.
        tag_w = text_width(text, TAG_FONTSIZE) + 2 * TAG_PAD
        cy = y + h / 2
        r = mpatches.FancyBboxPatch(
            (TAG_X, cy - TAG_H / 2), tag_w, TAG_H,
            boxstyle=f"round,pad=0.015,rounding_size={TAG_H / 2:.3f}",
            fc=LIGHT_PURPLE, ec="none", zorder=3,
        )
        ax.add_patch(r)
        ax.text(TAG_X + tag_w / 2, cy, text, ha="center", va="center",
                 color=DARK_PURPLE, fontsize=TAG_FONTSIZE, fontweight="bold",
                 zorder=4)

    # =========================================================================
    # Row 1: WSI -> tiles -> frozen encoder -> ABMIL
    # =========================================================================
    y0, h0 = 10.75, 1.35
    wsi_x = 0.2
    row_band(y0, h0)
    row_label(y0, h0, "MODEL TRAINING")

    frame = mpatches.FancyBboxPatch(
        (wsi_x, y0), ROW1_W, h0, boxstyle="round,pad=0.04,rounding_size=0.1",
        fc="white", ec=DARK_PURPLE, lw=1.0, zorder=3,
    )
    ax.add_patch(frame)
    if WSI_THUMB.exists():
        im = np.asarray(Image.open(WSI_THUMB).convert("RGB").resize((260, 260)))
        pad = 0.06
        img_side = h0 - 2 * pad
        img_x0 = wsi_x + (ROW1_W - img_side) / 2
        ax.imshow(im, extent=(img_x0, img_x0 + img_side, y0 + pad, y0 + h0 - pad),
                   zorder=4, aspect="auto")
    ax.text(wsi_x + ROW1_W / 2, y0 - 0.14, "H&E WSI", ha="center", va="top", fontsize=BOX_FONTSIZE,
            color=DARK_PURPLE, fontweight="bold")

    x = wsi_x + ROW1_W
    arrow(ax, (x, y0 + h0 / 2), (x + GAP, y0 + h0 / 2))
    x += GAP
    box(ax, x, y0, PREPROC_W, h0, "")  # frame only; image + grid drawn on top below
    tile_img_w = h0 - 0.2
    tile_img_x = x + 0.1
    tile_img_y = y0 + 0.1
    if WSI_THUMB.exists():
        im = np.asarray(Image.open(WSI_THUMB).convert("RGB").resize((200, 200)))
        ax.imshow(im, extent=(tile_img_x, tile_img_x + tile_img_w,
                               tile_img_y, tile_img_y + tile_img_w), zorder=4, aspect="auto")
        for i in range(1, 4):
            gx = tile_img_x + tile_img_w * i / 4
            ax.plot([gx, gx], [tile_img_y, tile_img_y + tile_img_w], color="white",
                    lw=0.6, zorder=5)
            gy = tile_img_y + tile_img_w * i / 4
            ax.plot([tile_img_x, tile_img_x + tile_img_w], [gy, gy], color="white",
                    lw=0.6, zorder=5)
        ax.add_patch(mpatches.Rectangle((tile_img_x, tile_img_y), tile_img_w, tile_img_w,
                                         fc="none", ec=DARK_PURPLE, lw=0.8, zorder=6))
    ax.text(tile_img_x + tile_img_w + 0.1, y0 + h0 / 2,
            "Preprocessing\n(tiled,\nfiltered,\nunstained)",
            ha="left", va="center", fontsize=BOX_FONTSIZE, linespacing=1.3, zorder=4)
    x += PREPROC_W
    arrow(ax, (x, y0 + h0 / 2), (x + GAP, y0 + h0 / 2))
    x += GAP
    box(ax, x, y0, ROW1_W, h0,
        "Pretrained\nencoder\n(3 variants)",
        fontsize=BOX_FONTSIZE, fc=LIGHT_PURPLE)
    x += ROW1_W
    arrow(ax, (x, y0 + h0 / 2), (x + GAP, y0 + h0 / 2))
    x += GAP
    box(ax, x, y0, ROW1_W, h0,
        f"ABMIL classifier\ntrained on\nEBRAINS (Europe)\nn={ef['n_final']}",
        fontsize=BOX_FONTSIZE, fc=MID_PURPLE)
    abmil_cx = x + ROW1_W / 2

    # =========================================================================
    # Row 2: external evaluation -- branch into two cohorts
    # =========================================================================
    ev_h = 1.1
    ev_y = y0 - ROW_GAP - ev_h
    row_label(ev_y, ev_h, "EXTERNAL VALIDATION")
    tcga_x, tcga_w = 1.7, 3.1
    ipd_x, ipd_w = tcga_x + tcga_w + GAP, 2.9
    tcga_cx, ipd_cx = tcga_x + tcga_w / 2, ipd_x + ipd_w / 2
    row12_bend = (y0 + ev_y + ev_h) / 2
    elbow_arrow(ax, abmil_cx, y0, tcga_cx, ev_y + ev_h, y_bend=row12_bend)
    elbow_arrow(ax, abmil_cx, y0, ipd_cx, ev_y + ev_h, y_bend=row12_bend)

    box(ax, tcga_x, ev_y, tcga_w, ev_h,
        f"TCGA (multi-country)\nn={tf['n_final']}",
        fontsize=BOX_FONTSIZE, fc=PALE_PURPLE)
    box(ax, ipd_x, ev_y, ipd_w, ev_h,
        f"IPD Brain (India)\nn={ipf['n_final']}",
        fontsize=BOX_FONTSIZE, fc=PALE_PURPLE)


    unc_h = 1.1
    unc_y = ev_y - ROW_GAP - unc_h
    row_band(unc_y, unc_h)
    row_label(unc_y, unc_h, "UNCERTAINTY ESTIMATION")
    unc_x, unc_w = 0.25, 5.5
    ood_x, ood_w = unc_x + unc_w + GAP, 1.8
    unc_cx, ood_cx = unc_x + unc_w / 2, ood_x + ood_w / 2

    bus2_y = ev_y - 0.2
    bus2_left = min(tcga_cx, unc_cx)
    bus2_right = max(ipd_cx, ood_cx)
    ax.plot([tcga_cx, tcga_cx], [ev_y, bus2_y], color=LINE_COLOR, lw=1.5, zorder=2,
            solid_joinstyle="round", solid_capstyle="round")
    ax.plot([ipd_cx, ipd_cx], [ev_y, bus2_y], color=LINE_COLOR, lw=1.5, zorder=2,
            solid_joinstyle="round", solid_capstyle="round")
    ax.plot([bus2_left, bus2_right], [bus2_y, bus2_y], color=LINE_COLOR, lw=1.5, zorder=2,
            solid_joinstyle="round", solid_capstyle="round")
    arrow(ax, (unc_cx, bus2_y), (unc_cx, unc_y + unc_h))
    arrow(ax, (ood_cx, bus2_y), (ood_cx, unc_y + unc_h))
    box(ax, unc_x, unc_y, unc_w, unc_h,
        "Five classifier-confidence signals\n"
        "predictive entropy · deep ensemble · MC-dropout ·\nLaplace · epistemic",
        fontsize=BOX_FONTSIZE, fc=LIGHT_PURPLE)
    box(ax, ood_x, unc_y, ood_w, unc_h,
        "OOD\ndetection\n(density-based)",
        fontsize=BOX_FONTSIZE, fc=LIGHT_PURPLE)


    dec_h = 1.1
    dec_y = unc_y - ROW_GAP - dec_h
    row_label(dec_y, dec_h, "DECISION MECHANISMS")
    con_x, con_w = 7.15, 3.15
    con_cx = con_x + con_w / 2
    MARGIN = 0.075
    sel_x = unc_cx - MARGIN
    sel_w = (con_x - GAP) - sel_x
    sel_cx = sel_x + sel_w / 2
    # Five-signals -> Selective is a plain vertical: unc_cx already falls
    # inside Selective's span, so no bend is needed at all.
    assert sel_x < unc_cx < sel_x + sel_w
    arrow(ax, (unc_cx, unc_y), (unc_cx, dec_y + dec_h))
    sel_entry, con_entry = sel_cx + 0.7, con_cx
    fork_y = (unc_y + dec_y + dec_h) / 2
    assert unc_y > fork_y > dec_y + dec_h
    ax.plot([ood_cx, ood_cx], [unc_y, fork_y], color=LINE_COLOR, lw=1.5, zorder=2,
            solid_joinstyle="round", solid_capstyle="round")
    ax.plot([sel_entry, con_entry], [fork_y, fork_y], color=LINE_COLOR, lw=1.5, zorder=2,
            solid_joinstyle="round", solid_capstyle="round")
    arrow(ax, (sel_entry, fork_y), (sel_entry, dec_y + dec_h))
    arrow(ax, (con_entry, fork_y), (con_entry, dec_y + dec_h))
    box(ax, sel_x, dec_y, sel_w, dec_h,
        "Selective prediction\nheuristic deferral",
        fontsize=BOX_FONTSIZE, fc=PALE_PURPLE)
    box(ax, con_x, dec_y, con_w, dec_h,
        "Conformal prediction\nformal guarantee",
        fontsize=BOX_FONTSIZE, fc=PALE_PURPLE)

    test_h = 1.1
    test_y = dec_y - ROW_GAP - test_h
    row_band(test_y, test_h)
    row_label(test_y, test_h, "ROBUSTNESS TESTING")
    # Equal-width sibling boxes (both hold a single short line, so both fit
    # comfortably at the same font size) separated by the standard GAP.
    fair_w = shift_w = 3.35
    fair_x = sel_cx - 1.7
    shift_x = fair_x + fair_w + GAP
    assert shift_x + shift_w <= 10.3, "Distribution-shift robustness box overflows canvas"
    sel_shift_entry = shift_x + 0.4
    assert sel_shift_entry < con_cx - 0.3, "selective->shift entry must stay clear of conformal's own line"
    arrow(ax, (sel_cx, dec_y), (sel_cx, test_y + test_h))
    elbow_arrow(ax, sel_cx, dec_y, sel_shift_entry, test_y + test_h, y_bend=dec_y - 0.22)
    arrow(ax, (con_cx, dec_y), (con_cx, test_y + test_h))
    box(ax, fair_x, test_y, fair_w, test_h, "Demographic fairness",
        fontsize=BOX_FONTSIZE, fc=PALE_PURPLE)
    box(ax, shift_x, test_y, shift_w, test_h, "Distribution-shift robustness",
        fontsize=BOX_FONTSIZE, fc=PALE_PURPLE)
    fair_shift_gap_mid = (fair_x + fair_w + shift_x) / 2

    # =========================================================================
    # Second biomarker -- separate parallel arm, dashed, reusing everything
    # =========================================================================
    branch_h = 1.35
    branch_y = test_y - ROW_GAP - branch_h
    box(ax, 1.6, branch_y, 7.3, branch_h,
        "Second biomarker: 1p/19q codeletion\n"
        "same pipeline throughout, different label",
        fontsize=BOX_FONTSIZE, fc="white", ec=BRIGHT_PURPLE, linestyle="dashed", shadow=False)
    arrow(ax, (fair_shift_gap_mid, test_y), (fair_shift_gap_mid, branch_y + branch_h),
          color=BRIGHT_PURPLE, ls="dashed")

    plt.savefig(OUT / "figure_study_design_overview.png")
    plt.savefig(OUT / "figure_study_design_overview.pdf")

    # pgf's text measurement genuinely invokes pdflatex, where a bare "&" is
    # an (invalid, here) alignment-tab character -- Agg/pdf don't hit this
    # since they never run real LaTeX. Escape only for the pgf save, then
    # restore, so the png/pdf keep the plain "H&E" the reader should see.
    originals = [t.get_text() for t in ax.texts]
    for t in ax.texts:
        t.set_text(t.get_text().replace("&", r"\&"))
    fig.savefig(OUT / "figure_study_design_overview.pgf", backend="pgf")
    for t, orig in zip(ax.texts, originals):
        t.set_text(orig)

    plt.close()
    print("Saved figure_study_design_overview.{png,pdf,pgf} to", OUT)


if __name__ == "__main__":
    main()
