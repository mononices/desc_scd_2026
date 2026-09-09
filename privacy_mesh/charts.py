import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from privacy_mesh.experiment import CHART_DIR

EPS_LABELS = {None: "∞", 8.0: "8", 4.0: "4", 2.0: "2", 1.0: "1"}

COLOR_UTILITY = "#1f77b4"
COLOR_ATTACK = "#d62728"
COLOR_CENTRAL = "#2ca02c"
COLOR_FL = "#ff7f0e"


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


def tradeoff_figure(records, highlight_key=None):
    items = sorted(records.values(), key=_order)
    xs = np.arange(len(items))
    ticks = [_labels(r) for r in items]
    accs = np.array([r["acc"] for r in items])
    acc_std = np.array([r.get("acc_std", 0.0) for r in items])
    atts = np.array([r["attack_auc"] for r in items])
    att_std = np.array([r.get("attack_auc_std", 0.0) for r in items])
    fig, ax1 = plt.subplots(figsize=(10.5, 5.4))
    ax1.set_ylabel("model utility — test accuracy", color=COLOR_UTILITY)
    ax1.set_ylim(0.55, 0.9)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(ticks, fontsize=9.5)
    ax1.errorbar(xs, accs, yerr=acc_std, fmt="-o", color=COLOR_UTILITY, lw=2, ms=7,
                 capsize=4, zorder=4, label="model accuracy (mean ± std)")
    ax2 = ax1.twinx()
    ax2.set_ylabel("privacy risk — membership-inference attack AUC", color=COLOR_ATTACK)
    ax2.set_ylim(0.35, 0.85)
    ax2.axhline(0.5, color="#999999", lw=1.2, ls=":")
    ax2.text(len(items) - 0.05, 0.507, "chance level (AUC = 0.5)", ha="right", va="bottom",
             fontsize=8, color="#666666")
    bars = ax2.bar(xs, atts, width=0.5, color=COLOR_ATTACK, alpha=0.7, zorder=3,
                   yerr=att_std, capsize=4, error_kw={"lw": 1.2, "ecolor": "#7a1414"},
                   label="attack AUC (mean ± std)")
    refs = []
    for rec in records.values():
        if rec["arch"] == "centralized":
            ax1.axhline(rec["acc"], color=COLOR_CENTRAL, ls="--", lw=1.3)
            ax2.axhline(rec["attack_auc"], color=COLOR_CENTRAL, ls="--", lw=1.3)
            refs.append(plt.Line2D([0], [0], color=COLOR_CENTRAL, ls="--", lw=1.3,
                                   label="centralised accuracy / attack AUC"))
        elif rec["arch"] == "federated":
            ax1.axhline(rec["acc"], color=COLOR_FL, ls=":", lw=1.4)
            ax2.axhline(rec["attack_auc"], color=COLOR_FL, ls=":", lw=1.4)
            refs.append(plt.Line2D([0], [0], color=COLOR_FL, ls=":", lw=1.4,
                                   label="federated (no DP) accuracy / attack AUC"))
    if highlight_key:
        for r in items:
            if r["key"] == highlight_key:
                ax1.scatter([_order(r)], [r["acc"]], s=190, facecolors="none",
                            edgecolors="#111111", linewidths=1.8, zorder=6)
                for bar in bars:
                    if abs(bar.get_x() + bar.get_width() / 2 - _order(r)) < 1e-9:
                        bar.set_edgecolor("#111111")
                        bar.set_linewidth(1.8)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2 + refs, labels1 + labels2 + [r.get_label() for r in refs],
               loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Privacy vs utility — federated learning with differential privacy (δ = 1e-5)",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
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


def save_figure(fig, name):
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
