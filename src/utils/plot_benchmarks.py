"""
MFGF TVPR Training Benchmark Plotting Utilities
Generates professional benchmark charts:
  1. Multi-Loss Convergence Curve (L_total, L_common, L_d2)
  2. Dynamic Alpha Blending & Learning Rate Evolution (Paper Figure 5)
  3. CMC Cumulative Matching Characteristics (Rank@1, Rank@5, Rank@10, Rank@50)
  4. Stability & Quality Metrics (Median Rank - MdR, mAP, MRR)
"""

import os
import json
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Colab & servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .paths import CHECKPOINTS_DIR, REPORTS_DIR


def save_training_history(history: Dict, save_path: Optional[str] = None):
    """Saves training history to JSON file."""
    if save_path is None:
        save_path = str(CHECKPOINTS_DIR / "train_history.json")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_training_history(save_path: Optional[str] = None) -> Optional[Dict]:
    """Loads training history from JSON file if it exists."""
    if save_path is None:
        save_path = str(CHECKPOINTS_DIR / "train_history.json")
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Warning: Could not read {save_path}: {e}")
    return None


def plot_training_dashboard(
    history: Dict,
    output_path: Optional[str] = None,
    dpi: int = 200,
    title_suffix: str = ""
) -> str:
    """
    Plots a 4-in-1 comprehensive Training Benchmark Dashboard.
    Returns the absolute path of the generated image.
    """
    if output_path is None:
        output_path = str(REPORTS_DIR / "training_dashboard.png")
    os.makedirs(os.path.dirname(output_path) or "reports", exist_ok=True)

    epochs = history.get("epochs", [])
    if not epochs:
        print("[!] No epoch data to plot.")
        return output_path

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")

    title = f"MFGF TVPR — Training & Evaluation Benchmark Dashboard{title_suffix}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98, color="#1e1e2e")

    # -------------------------------------------------------------------------
    # Subplot 1: Loss Convergence Curves (Top-Left)
    # -------------------------------------------------------------------------
    ax_loss = axes[0, 0]
    ax_loss.plot(epochs, history.get("loss", []), label=r"Total Loss $\mathcal{L}_{MFGF}$", color="#d20f39", lw=2.2)
    if "loss_common" in history and history["loss_common"]:
        ax_loss.plot(epochs, history["loss_common"], label=r"Common Space $\mathcal{L}_{common}$ (InfoNCE)", color="#1e66f5", lw=1.8, linestyle="--")
    if "loss_d2" in history and history["loss_d2"]:
        ax_loss.plot(epochs, history["loss_d2"], label=r"Dual-Distilled $\mathcal{L}_{D2}$ (Tips)", color="#40a02b", lw=1.8, linestyle=":")

    ax_loss.set_title("1. Loss Convergence Curves", fontsize=12, fontweight="bold", color="#2c3e50")
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Loss", fontsize=10)
    ax_loss.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # -------------------------------------------------------------------------
    # Subplot 2: Dynamic Alpha & Learning Rate Schedule (Top-Right)
    # -------------------------------------------------------------------------
    ax_alpha = axes[0, 1]
    ax_lr = ax_alpha.twinx()

    alpha_vals = history.get("alpha", [])
    if alpha_vals:
        line_alpha = ax_alpha.plot(epochs, alpha_vals, label=r"Dynamic Blending $\alpha$", color="#8839ef", lw=2.4)
        ax_alpha.set_ylabel(r"Dynamic Alpha ($\alpha$)", color="#8839ef", fontsize=10, fontweight="bold")
        ax_alpha.tick_params(axis="y", labelcolor="#8839ef")
        # Bounds reference
        ax_alpha.axhline(0.15, color="#8839ef", linestyle=":", alpha=0.4, label=r"$\alpha_{min}=0.15$")
        ax_alpha.axhline(0.85, color="#8839ef", linestyle=":", alpha=0.4, label=r"$\alpha_{max}=0.85$")
        ax_alpha.set_ylim(0.0, 1.0)

    lr_vals = history.get("lr", [])
    if lr_vals:
        line_lr = ax_lr.plot(epochs, lr_vals, label="Learning Rate (Cosine)", color="#e64553", lw=1.6, linestyle="--")
        ax_lr.set_ylabel("Learning Rate", color="#e64553", fontsize=10, fontweight="bold")
        ax_lr.tick_params(axis="y", labelcolor="#e64553")
        ax_lr.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))

    ax_alpha.set_title("2. Dynamic Alpha Blending & Learning Rate Schedule", fontsize=12, fontweight="bold", color="#2c3e50")
    ax_alpha.set_xlabel("Epoch", fontsize=10)
    ax_alpha.grid(True, linestyle="--", alpha=0.5)
    ax_alpha.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Combined legend for twinx
    lines = (line_alpha if alpha_vals else []) + (line_lr if lr_vals else [])
    labels = [l.get_label() for l in lines]
    if lines:
        ax_alpha.legend(lines, labels, loc="center right", frameon=True, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Subplot 3: CMC Accuracy Progression (Rank@1, 5, 10, 50) (Bottom-Left)
    # -------------------------------------------------------------------------
    ax_cmc = axes[1, 0]
    eval_epochs = history.get("eval_epochs", [])

    if eval_epochs and "rank1" in history and history["rank1"]:
        r1 = history["rank1"]
        r5 = history.get("rank5", [])
        r10 = history.get("rank10", [])
        r50 = history.get("rank50", [])

        ax_cmc.plot(eval_epochs, r1, marker="o", label="Rank@1", color="#d20f39", lw=2.2)
        if r5:
            ax_cmc.plot(eval_epochs, r5, marker="s", label="Rank@5", color="#fe640b", lw=2.0)
        if r10:
            ax_cmc.plot(eval_epochs, r10, marker="^", label="Rank@10", color="#40a02b", lw=1.8)
        if r50:
            ax_cmc.plot(eval_epochs, r50, marker="D", label="Rank@50", color="#179299", lw=1.8)

        # Highlight best Rank@1
        best_r1 = max(r1)
        best_ep = eval_epochs[r1.index(best_r1)]
        ax_cmc.annotate(
            f"Best R@1: {best_r1:.1f}% (@Ep{best_ep})",
            xy=(best_ep, best_r1),
            xytext=(best_ep, min(100.0, best_r1 + 6.0)),
            arrowprops=dict(facecolor="#d20f39", shrink=0.08, width=1.5, headwidth=6),
            fontweight="bold",
            color="#d20f39",
            fontsize=9
        )
        ax_cmc.set_ylim(0.0, 105.0)
    else:
        ax_cmc.text(0.5, 0.5, "Evaluation metrics will appear\nafter first evaluation epoch", ha="center", va="center", color="#7c7f93", fontsize=11)

    ax_cmc.set_title("3. Cumulative Matching Characteristics (CMC Accuracy)", fontsize=12, fontweight="bold", color="#2c3e50")
    ax_cmc.set_xlabel("Epoch", fontsize=10)
    ax_cmc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_cmc.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax_cmc.grid(True, linestyle="--", alpha=0.5)
    ax_cmc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # -------------------------------------------------------------------------
    # Subplot 4: System Stability & Precision (MdR, mAP, MRR) (Bottom-Right)
    # -------------------------------------------------------------------------
    ax_stab = axes[1, 1]
    ax_map = ax_stab.twinx()

    if eval_epochs and "mdr" in history and history["mdr"]:
        mdr_vals = history["mdr"]
        line_mdr = ax_stab.plot(eval_epochs, mdr_vals, marker="v", label="Median Rank (MdR) ↓", color="#df8e1d", lw=2.2)
        ax_stab.set_ylabel("Median Rank (MdR) [Lower is Better]", color="#df8e1d", fontsize=10, fontweight="bold")
        ax_stab.tick_params(axis="y", labelcolor="#df8e1d")

        # Highlight lowest MdR
        valid_mdr = [m for m in mdr_vals if m != float("inf")]
        if valid_mdr:
            min_mdr = min(valid_mdr)
            min_ep = eval_epochs[mdr_vals.index(min_mdr)]
            ax_stab.annotate(
                f"Lowest MdR: {min_mdr:.1f} (@Ep{min_ep})",
                xy=(min_ep, min_mdr),
                xytext=(min_ep, min_mdr + max(2.0, min_mdr * 0.2)),
                arrowprops=dict(facecolor="#df8e1d", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold",
                color="#df8e1d",
                fontsize=9
            )

        map_vals = history.get("map", [])
        mrr_vals = history.get("mrr", [])
        lines_right = []
        if map_vals:
            l_map = ax_map.plot(eval_epochs, map_vals, marker="o", label="mAP (%) ↑", color="#209fb5", lw=2.0, linestyle="--")
            lines_right.append(l_map[0])
        if mrr_vals:
            l_mrr = ax_map.plot(eval_epochs, mrr_vals, marker="*", label="MRR (%) ↑", color="#04a5e5", lw=1.6, linestyle=":")
            lines_right.append(l_mrr[0])

        ax_map.set_ylabel("Precision / Rank (%) [Higher is Better]", color="#209fb5", fontsize=10, fontweight="bold")
        ax_map.tick_params(axis="y", labelcolor="#209fb5")
        ax_map.set_ylim(0.0, 105.0)

        # Legend
        all_l = line_mdr + lines_right
        ax_stab.legend(all_l, [l.get_label() for l in all_l], loc="center right", frameon=True, framealpha=0.9)
    else:
        ax_stab.text(0.5, 0.5, "Stability metrics will appear\nafter first evaluation epoch", ha="center", va="center", color="#7c7f93", fontsize=11)

    ax_stab.set_title("4. Stability & Quality Metrics (MdR & mAP/MRR)", fontsize=12, fontweight="bold", color="#2c3e50")
    ax_stab.set_xlabel("Epoch", fontsize=10)
    ax_stab.grid(True, linestyle="--", alpha=0.5)
    ax_stab.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.08, right=0.92, hspace=0.32, wspace=0.28)
    plt.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[+] Saved Training Benchmark Dashboard to: {os.path.abspath(output_path)}", flush=True)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot MFGF Training Benchmarks from history JSON")
    parser.add_argument("--history", type=str, default=str(CHECKPOINTS_DIR / "train_history.json"), help="Path to train_history.json")
    parser.add_argument("--output", type=str, default=str(REPORTS_DIR / "training_dashboard.png"), help="Output image file path")
    args = parser.parse_args()

    data = load_training_history(args.history)
    if data:
        plot_training_dashboard(data, output_path=args.output)
    else:
        print(f"[!] History file '{args.history}' not found.")
