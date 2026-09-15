"""Generate professional method diagrams (PNG) for the thesis report."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# Restrained, professional palette
INK = "#1b2a41"        # near-black ink for text/borders
STEEL = "#2f5c8f"      # primary
STEEL_L = "#eaf0f7"    # light panel fill
CARD = "#f4f7fb"       # agent card fill
JUDGE = "#8a6d3b"      # adjudicator accent
JUDGE_L = "#f6efe1"
OUT_G = "#2f6b46"      # output accent
OUT_L = "#e6f0ea"
GATE = "#b98a2e"
GREY = "#5a6472"


def box(ax, x, y, w, h, text, fc, ec=INK, fs=10, tc=INK, bold=False, r=0.015, lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={r}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=4, fontweight="bold" if bold else "normal")


def pill(ax, x, y, w, h, text, fc, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.03",
                                linewidth=0, facecolor=fc, zorder=5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2,
            color=tc, zorder=6, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 lw=lw, color=color, zorder=2))


def band(ax, y, h, label):
    ax.add_patch(FancyBboxPatch((0.5, y), 99, h, boxstyle="round,pad=0,rounding_size=0",
                                linewidth=0, facecolor="none", zorder=0))
    ax.text(2.0, y + h / 2, label, ha="left", va="center", fontsize=8.2,
            color=STEEL, fontweight="bold", rotation=90)


def architecture():
    fig, ax = plt.subplots(figsize=(11.5, 8.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(52, 97.5, "Figure 1. The proposed multi-agent clinical-reasoning framework",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    # left stage rail
    for y, h, lab in [(80, 9, "INPUT"), (55, 22, "STAGE 1\nDiverse panel"),
                      (42, 9, "STAGE 2\nAggregation"), (20, 18, "STAGE 3\nAdjudication"),
                      (5, 9, "OUTPUT")]:
        band(ax, y, h, lab)

    # Input
    box(ax, 34, 81, 32, 7.5, "Clinical MCQ\nstem + options (A–D)", STEEL_L, ec=STEEL, fs=10.5, bold=True)

    # Panel container
    ax.add_patch(FancyBboxPatch((7.5, 55.5), 85, 21, boxstyle="round,pad=0.01,rounding_size=0.02",
                                linewidth=1.1, edgecolor=STEEL, facecolor="#fbfcfe", linestyle=(0, (5, 3)), zorder=1))
    ax.text(50, 74.6, "Diverse Reasoning Panel  —  four independent agents in parallel (temperature 0)",
            ha="center", fontsize=8.6, color=STEEL, style="italic", zorder=2)
    agents = [("Agent A\nDirect diagnosis", 10, "GPT", STEEL),
              ("Agent B\nElimination", 33, "GPT", STEEL),
              ("Agent C\nPathophysiology", 56, "GPT", STEEL),
              ("Agent D\nSecond opinion", 79, "Claude", JUDGE)]
    for label, x, prov, pc in agents:
        box(ax, x, 58.5, 18.5, 11.5, label, CARD, ec=INK, fs=9)
        pill(ax, x + 4.5, 66.6, 9.5, 2.6, prov, pc)
        arrow(ax, 50, 81, x + 9.25, 70.2, color=GREY, lw=1.2)
    ax.text(50, 57.0, "each agent emits:  answer letter  +  confidence  +  concise rationale",
            ha="center", fontsize=7.8, color=GREY, style="italic", zorder=4)

    # Aggregation
    box(ax, 30, 43, 40, 7.5, "Confidence-weighted aggregation", STEEL_L, ec=STEEL, fs=10.5, bold=True)
    for _, x, _, _ in agents:
        arrow(ax, x + 9.25, 58.5, 50, 50.5, color=GREY, lw=1.1)

    # Decision gate (clean chevron/diamond)
    cx, cy = 50, 34
    ax.add_patch(Polygon([[cx, cy + 6], [cx + 16, cy], [cx, cy - 6], [cx - 16, cy]], closed=True,
                         facecolor="#fdf6e6", edgecolor=GATE, lw=1.4, zorder=3))
    ax.text(cx, cy, "Do the agents\nreach consensus?", ha="center", va="center",
            fontsize=8.8, color="#5c4410", fontweight="bold", zorder=4)
    arrow(ax, 50, 43, 50, 40, color=GREY)

    # Consensus branch (left)
    box(ax, 6, 22, 24, 8, "Yes — adopt the\nweighted consensus", OUT_L, ec=OUT_G, fs=9, tc="#20402e")
    arrow(ax, 34, 34, 30, 27, color=OUT_G); ax.text(30.5, 31.5, "consensus", fontsize=7.8, color=OUT_G)

    # Judge branch (right)
    box(ax, 70, 21.5, 26, 10, "No — Adjudicating Judge\nreviews every agent's\nletter + rationale and\nre-decides (forced choice)",
        JUDGE_L, ec=JUDGE, fs=8.6, tc="#4a3a1c")
    arrow(ax, 66, 34, 70, 28, color=JUDGE); ax.text(66.5, 31.5, "disagreement", fontsize=7.8, color=JUDGE)
    pill(ax, 79, 30.2, 8.5, 2.4, "GPT", JUDGE)

    # Output
    box(ax, 33, 6, 34, 8, "Final forced-choice answer", OUT_G, ec=OUT_G, fs=11, tc="white", bold=True)
    arrow(ax, 18, 22, 40, 14, color=OUT_G)
    arrow(ax, 83, 21.5, 60, 14, color=JUDGE)

    # legend
    ax.text(50, 1.7, "Provider assignment is fixed per agent by a deterministic hash; the judge is invoked "
            "only on disagreement (~15–25% of cases).", ha="center", fontsize=7.8, color=GREY, style="italic")
    fig.tight_layout(); fig.savefig(OUT / "fig_architecture.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def protocol():
    fig, ax = plt.subplots(figsize=(11.5, 6.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 96, "Figure 2. Experimental protocol and generalizability design", ha="center",
            fontsize=12.5, fontweight="bold", color=INK)

    box(ax, 4, 78, 44, 13, "Dataset 1 — MedMCQA-style\ngold recovered by LLM extraction\nwith deterministic cross-check", STEEL_L, ec=STEEL, fs=9)
    box(ax, 52, 78, 44, 13, "Dataset 2 — MedQA-USMLE\nbuilt-in answer key\n(external generalizability test)", JUDGE_L, ec=JUDGE, fs=9, tc="#4a3a1c")

    box(ax, 9, 60, 34, 8, "Development split\ntune adjudication policy", STEEL_L, ec=STEEL, fs=9)
    box(ax, 9, 45, 34, 8, "Freeze policy, prompts & models", OUT_L, ec=OUT_G, fs=9.5, tc="#20402e", bold=True)
    box(ax, 9, 30, 34, 8, "Held-out TEST split\nsingle evaluation", STEEL_L, ec=STEEL, fs=9)
    arrow(ax, 26, 78, 26, 68); arrow(ax, 26, 60, 26, 53); arrow(ax, 26, 45, 26, 38)

    box(ax, 58, 33, 36, 12, "Apply frozen framework\nwithout any retuning\nsingle evaluation", JUDGE_L, ec=JUDGE, fs=9.2, tc="#4a3a1c")
    arrow(ax, 74, 78, 74, 45, color=JUDGE)
    arrow(ax, 43, 49, 58, 41, color=OUT_G, style="-|>"); ax.text(50.5, 52, "same frozen\nframework", ha="center", fontsize=7.4, color=OUT_G)

    box(ax, 24, 11, 52, 9, "Performance + safety tables, visualizations,\nand paired McNemar tests on both datasets", INK, ec=INK, fs=9.4, tc="white", bold=True)
    arrow(ax, 26, 30, 42, 20); arrow(ax, 76, 33, 60, 20, color=JUDGE)

    ax.text(50, 4.2, "Systems compared: Single-agent GPT  |  Single-agent Claude  |  Multi-Agent framework (proposed)",
            ha="center", fontsize=8.4, color=GREY, style="italic")
    fig.tight_layout(); fig.savefig(OUT / "fig_protocol.png", dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    architecture(); protocol()
    print("Wrote diagrams to", OUT)
