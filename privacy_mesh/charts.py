import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from privacy_mesh.experiment import CHART_DIR

EPS_LABELS = {None: "∞", 8.0: "8", 4.0: "4", 2.0: "2", 1.0: "1"}

COLOR_UTILITY = "#2a78d6"
COLOR_ATTACK = "#e34948"
COLOR_CENTRAL = "#1baf7a"
COLOR_FL = "#eb6834"
COLOR_GRID = "#d8d8d4"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#52514e"


def _order(record):
    if record["arch"] == "centralized":
        return 0
    if record["arch"] == "federated":
        return 1
    return {8.0: 2, 4.0: 3, 2.0: 4, 1.0: 5}.get(record["eps_target"], 6)


def _labels(record):
    if record["arch"] == "centralized":
        return "centralised\n(pooled data)"
    if record["arch"] == "federated":
        return "federated\n(no DP)"
    return f"ε = {EPS_LABELS[record['eps_target']]}"


def _tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9.5, length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLOR_GRID, lw=0.8, alpha=0.7)


def tradeoff_figure(records, highlight_key=None):
    items = sorted(records.values(), key=_order)
    xs = np.arange(len(items))
    ticks = [_labels(r) for r in items]
    accs = np.array([r["acc"] for r in items])
    acc_std = np.array([r.get("acc_std", 0.0) for r in items])
    atts = np.array([100.0 * r["attack_acc"] for r in items])
    att_std = np.array([100.0 * r.get("attack_acc_std", 0.0) for r in items])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.0, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18},
    )

    ax1.errorbar(xs, accs, yerr=acc_std, fmt="-o", color=COLOR_UTILITY, lw=2, ms=8,
                 capsize=4, zorder=4, markeredgecolor="white", markeredgewidth=1.4)
    for x, a, e in zip(xs, accs, acc_std):
        ax1.annotate(f"{a:.3f}", (x, a + e), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=9, color=COLOR_INK)
    ax1.set_ylabel("test accuracy", fontsize=10, color=COLOR_MUTED)
    ax1.set_ylim(min(accs) - 0.06, max(accs) + 0.055)
    ax1.set_title("Model utility — higher is better", fontsize=11, color=COLOR_INK,
                  loc="left", pad=8)
    _tidy(ax1)

    bars = ax2.bar(xs, atts - 50.0, bottom=50.0, width=0.56, color=COLOR_ATTACK, zorder=3,
                   yerr=att_std, capsize=4, error_kw={"lw": 1.2, "ecolor": "#7a1414"})
    for bar in bars:
        bar.set_linewidth(2)
        bar.set_edgecolor("white")
    ax2.axhline(50.0, color=COLOR_MUTED, lw=1.4, zorder=5)
    ax2.text(-0.42, 49.6, "50% = the attacker is guessing  ·  bars show how far above chance",
             ha="left", va="top", fontsize=9.5, color=COLOR_MUTED)
    for x, a, e in zip(xs, atts, att_std):
        top = max(a, 50.0)
        ax2.annotate(f"{a:.1f}%", (x, top + e), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=9.5, color=COLOR_INK)
    ax2.set_ylabel("attack success rate", fontsize=10, color=COLOR_MUTED)
    ax2.set_ylim(47.5, max(atts) + 6)
    ax2.set_title("Privacy risk — how often the attacker correctly identifies a member",
                  fontsize=11, color=COLOR_INK, loc="left", pad=8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(ticks, fontsize=10, color=COLOR_INK)
    _tidy(ax2)

    if highlight_key:
        for i, r in enumerate(items):
            if r["key"] == highlight_key:
                ax1.scatter([i], [r["acc"]], s=230, facecolors="none",
                            edgecolors=COLOR_INK, linewidths=1.8, zorder=6)
                bars[i].set_edgecolor(COLOR_INK)

    fig.suptitle("Privacy costs almost nothing here — until it does",
                 fontsize=14, y=0.975, x=0.5, color=COLOR_INK)
    fig.text(0.5, 0.925,
             "Federated learning + DP-SGD across three UAE health authorities  ·  "
             "3 seeds per configuration  ·  δ = 1e-5",
             ha="center", fontsize=9.5, color=COLOR_MUTED)
    fig.tight_layout(rect=(0, 0.01, 1, 0.91))
    return fig


def roc_figure(record, reference=None):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fpr = np.asarray(record["roc_fpr"])
    tpr = np.asarray(record["roc_tpr"])
    ax.plot(fpr, tpr, color=COLOR_ATTACK, lw=2,
            label=f"{record['arch_label']} (AUC {record['attack_auc']:.3f})")
    if reference is not None:
        rfpr = np.asarray(reference["roc_fpr"])
        rtpr = np.asarray(reference["roc_tpr"])
        ax.plot(rfpr, rtpr, color=COLOR_CENTRAL, lw=1.6, ls="--",
                label=f"centralised baseline (AUC {reference['attack_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", lw=1, ls=":")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Membership-inference attack — ROC curve")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def confidence_figure(record, reference=None):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), sharey=True)
    for ax in axes:
        ax.set_xlabel("model confidence assigned to the record's true label")
        ax.set_ylabel("share of attack candidates")
        ax.set_xlim(0.5, 1.0)
    bins = np.linspace(0.5, 1.0, 31)
    mem = np.asarray(record["conf_members"])
    non = np.asarray(record["conf_nonmembers"])
    ax = axes[0]
    ax.hist(mem, bins=bins, density=True, alpha=0.7, color=COLOR_ATTACK)
    ax.hist(non, bins=bins, density=True, alpha=0.45, color=COLOR_UTILITY)
    ax.set_title(record["arch_label"])
    if reference is not None:
        ax = axes[1]
        rmem = np.asarray(reference["conf_members"])
        rnon = np.asarray(reference["conf_nonmembers"])
        ax.hist(rmem, bins=bins, density=True, alpha=0.7, color=COLOR_ATTACK)
        ax.hist(rnon, bins=bins, density=True, alpha=0.45, color=COLOR_UTILITY)
        ax.set_title("centralised baseline (for comparison)")
    axes[0].legend(["training members", "non-members"], fontsize=8.5)
    fig.suptitle("Do individuals in the training data look different to the model?",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def curve_figure(record):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    hist = np.asarray(record["history"])
    if record["arch"] == "centralized":
        xlabel = "training epoch"
    else:
        xlabel = "aggregation round"
    ax.plot(hist[:, 0], hist[:, 1], "-o", color=COLOR_UTILITY, lw=1.8, label="test accuracy")
    ax.plot(hist[:, 0], hist[:, 2], "-s", color="#9467bd", lw=1.4, label="test AUC")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("score")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(f"{record['arch_label']} — training progress")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def aia_figure(record, reference=None):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    true_vals = np.asarray(record.get("aia_true_sample", []))
    pred_vals = np.asarray(record.get("aia_pred_sample", []))
    all_vals = [true_vals, pred_vals]
    if len(true_vals) > 0:
        ax.scatter(true_vals, pred_vals, color=COLOR_ATTACK, alpha=0.7,
                   label=f"{record['arch_label']} (MAE {record.get('aia_mae', 0):.1f})")
    if reference is not None:
        rtrue = np.asarray(reference.get("aia_true_sample", []))
        rpred = np.asarray(reference.get("aia_pred_sample", []))
        if len(rtrue) > 0:
            ax.scatter(rtrue, rpred, color=COLOR_CENTRAL, alpha=0.35, marker="x",
                       label=f"centralised baseline (MAE {reference.get('aia_mae', 0):.1f})")
            all_vals += [rtrue, rpred]
    nonempty = [v for v in all_vals if len(v) > 0]
    if nonempty:
        min_v = min(v.min() for v in nonempty)
        max_v = max(v.max() for v in nonempty)
        ax.plot([min_v, max_v], [min_v, max_v], color=COLOR_MUTED, ls="--",
                label="perfect reconstruction")
    ax.set_xlabel("actual cholesterol (mg/dL)")
    ax.set_ylabel("inferred cholesterol (mg/dL)")
    ax.set_title(f"Attribute-inference attack — MAE {record.get('aia_mae', 0):.1f} mg/dL",
                 fontsize=11, color=COLOR_INK)
    ax.legend(loc="upper left", fontsize=8.5)
    _tidy(ax)
    fig.tight_layout()
    return fig


def save_figure(fig, name):
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
