import gradio as gr
import pandas as pd

from privacy_mesh import charts, experiment as exp

DESC = """
## UAE Privacy Mesh  -  privacy-preserving analytics across government health entities

Three emirate health authorities hold heart-screening records of **synthetic citizens**
(calibrated on the public UCI *Heart Disease* dataset). They are **not allowed to share raw
records**  -  instead each entity trains a local model and shares only protected parameter
updates with the aggregator.

Select an architecture and a privacy budget, run the experiment, then watch the red team try
to break it: a **membership-inference attack** asks *"was this individual in the training
data?"*, and an **attribute-inference attack** asks *"what was this individual's cholesterol
reading?"* -- both over a pool of true members and comparable non-members.

**Trust model.** The aggregator is assumed honest-but-curious: it receives only clipped,
optionally noised model updates, never raw records, and does not collude with any entity.
It is not protected against a malicious aggregator inspecting individual updates before
averaging -- that requires secure aggregation, which is complementary and out of scope here.

**Reading ε.** ε bounds how much any single person's record could have changed the
published model. There is no universal "safe" ε; as a reference point, production
systems that publish DP statistics from on-device or aggregate data typically operate
with ε roughly in the 2-16 range depending on sensitivity and query volume. The ε ≈ 7.4
budget used by default here sits within that band: strict enough to erase the
membership-inference and attribute-inference signal measured below, while remaining
accurate enough to be usable.
"""

HEADER = "# 🇦🇪 UAE Privacy Mesh"

ARCH_CHOICES = {
    "Centralised (pooled, unregularised)": "centralized",
    "Centralised (pooled, L2-regularised)": "centralized_reg",
    "Federated (FedAvg, unregularised)": "federated",
    "Federated (FedAvg, L2-regularised)": "federated_reg",
    "Federated + DP-SGD": "fed_dp",
}
EPS_CHOICES = {"∞ (no DP)": None, "8": 8.0, "4": 4.0, "2": 2.0, "1": 1.0}


def _fmt(value, digits=4):
    if value is None or isinstance(value, str):
        return "-"
    return f"{value:.{digits}f}"


def _metrics_df(record):
    n = record.get("n_reps", 1)
    spread = " (mean ± std over %d runs)" % n if n > 1 else ""
    eps_per = record.get("eps_per_client")
    sig_per = record.get("sigma_per_client")
    steps_per = record.get("steps_per_round")
    total_steps = record.get("total_steps")
    rows = [
        ("Architecture", record["arch_label"]),
        ("L2 weight decay (fair-baseline control)", f"{record.get('weight_decay', 0.0):g}"),
        ("Privacy budget ε (per entity)", "∞" if record["eps_target"] is None else f"{record['eps_target']:g}"),
        ("Achieved ε (Rényi DP accounting)", "∞" if record["eps_achieved"] is None
         else _fmt(record["eps_achieved"], 2)),
        ("Achieved ε per entity", "-" if not eps_per else ", ".join(f"{e:.2f}" for e in eps_per)),
        ("Noise multiplier σ per entity", "-" if not sig_per else ", ".join(f"{s:.2f}" for s in sig_per)),
        ("δ (failure probability)", f"{record['delta']:g}"),
        ("DP-SGD steps per round, per entity", "-" if not steps_per else ", ".join(str(s) for s in steps_per)),
        ("Total DP-SGD steps, per entity (rounds × steps/round)",
         "-" if not total_steps else ", ".join(str(s) for s in total_steps)),
        (" -  utility  - ", ""),
        ("Test accuracy" + spread, _fmt(record["acc"])),
        ("Test AUC", _fmt(record["auc"])),
        ("Training-set accuracy", _fmt(record.get("train_acc"))),
        ("Generalisation gap (train − test)", _fmt(record.get("gap"))),
        (" -  privacy risk  - ", ""),
        ("Attack accuracy (50% = chance)", _fmt(record["attack_acc"])),
        ("Attack AUC (membership inference)" + spread, _fmt(record["attack_auc"])),
        ("Attack TPR @ 1% FPR (1% = chance)", _fmt(record.get("tpr_at_1pct_fpr"))),
        ("Attack candidate pool (balanced)", str(record["n_candidates"])),
        (" -  run  - ", ""),
        ("Training records (all entities)", str(record["n_train"])),
        ("Aggregation rounds", str(record["rounds"])),
        ("Mean runtime per run", f"{record['time_s']} s"),
        (" -  attribute-inference attack (cholesterol)  - ", ""),
        ("AIA MAE (lower = higher risk)",
         "-" if record.get("aia_mae") is None else _fmt(record.get("aia_mae"), 1) + " mg/dL"),
        ("AIA baseline MAE (population-prior guess)",
         "-" if record.get("aia_baseline_mae") is None else _fmt(record.get("aia_baseline_mae"), 1) + " mg/dL"),
        ("AIA leak score (0 = no leak, 1 = perfect recon.)", _fmt(record.get("aia_leak_score"), 3)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _render(summaries, highlight=None):
    files = {"trade": None}
    if not summaries:
        return files
    fig = charts.tradeoff_figure(summaries, highlight_key=highlight)
    files["trade"] = charts.save_figure(fig, "tradeoff.png")
    if highlight and highlight in summaries:
        rec = summaries[highlight]
        central = summaries.get("centralized__none")
        fig = charts.roc_figure(rec, reference=central)
        files["roc"] = charts.save_figure(fig, f"roc_{highlight}.png")
        fig = charts.confidence_figure(rec, reference=central)
        files["conf"] = charts.save_figure(fig, f"conf_{highlight}.png")
        fig = charts.curve_figure(rec)
        files["curve"] = charts.save_figure(fig, f"curve_{highlight}.png")
        fig = charts.aia_figure(rec, reference=central)
        files["aia"] = charts.save_figure(fig, f"aia_{highlight}.png")
    return files


def _outputs(summaries, highlight):
    files = _render(summaries, highlight)
    out = [files["trade"]]
    if highlight and highlight in summaries:
        out += [_metrics_df(summaries[highlight]), files["roc"], files["conf"], files["curve"], files["aia"]]
    else:
        out += [None, None, None, None, None]
    return out


def _run_one(arch, eps, progress=gr.Progress()):
    base = exp.result_key(arch, eps)
    group = exp.group_records().get(base, [])
    rep = max((r["rep"] for r in group), default=-1) + 1

    def cb(frac, msg):
        progress(frac, desc=f"run {rep + 1}  -  {msg}")

    try:
        exp.run_experiment(arch, eps, rep=rep, progress=cb)
    except Exception as exc:
        raise gr.Error(f"experiment failed: {exc}")
    summaries = exp.config_summaries()
    return _outputs(summaries, base)


def _run_experiment(arch, eps_choice, progress=gr.Progress()):
    return _run_one(ARCH_CHOICES[arch], EPS_CHOICES[eps_choice], progress)


def _sweep_all(progress=gr.Progress()):
    groups = exp.group_records()
    todo = []
    for arch, eps in exp.sweep_configs():
        base = exp.result_key(arch, eps)
        have = max((r["rep"] for r in groups.get(base, [])), default=-1) + 1
        for rep in range(have, exp.PREWARM_REPS):
            todo.append((arch, eps, rep))
    total = max(1, len(todo))
    for i, (arch, eps, rep) in enumerate(todo):
        def cb(frac, msg, arch=arch, eps=eps, rep=rep, i=i):
            progress((i + frac) / total, f"[{arch} ε={eps}] run {rep + 1}  -  {msg}")

        exp.run_experiment(arch, eps, rep=rep, progress=cb)
    summaries = exp.config_summaries()
    return _outputs(summaries, None)


def _epsilon_update(arch):
    if arch == "Federated + DP-SGD":
        return gr.Dropdown(interactive=True, value="8")
    return gr.Dropdown(interactive=False, value="∞ (no DP)")


def build_app():
    summaries = exp.config_summaries()
    initial = _render(summaries, None)
    with gr.Blocks(title="UAE Privacy Mesh") as demo:
        gr.Markdown(HEADER)
        gr.Markdown(DESC)
        with gr.Row():
            with gr.Column(scale=1):
                arch = gr.Radio(
                    list(ARCH_CHOICES.keys()),
                    value="Centralised (pooled, unregularised)",
                    label="Architecture",
                )
                eps = gr.Dropdown(
                    list(EPS_CHOICES.keys()),
                    value="∞ (no DP)",
                    label="Privacy budget ε",
                    info="per-entity budget; applies to Federated + DP-SGD",
                    interactive=False,
                )
                run_btn = gr.Button("▶  Run experiment", variant="primary")
                sweep_btn = gr.Button("⚙  Complete comparison (3 runs per configuration)")
                gr.Markdown(
                    "**Reading the attack:** AUC = 0.5 means the attacker is guessing at chance "
                    "level. Values above 0.5 reveal that training membership leaks through the "
                    "model's confidence. **L2-regularised** rows are the fair baseline for "
                    "judging DP's accuracy cost, since the plain unregularised rows are "
                    "deliberately left vulnerable to make the naive attack demo work."
                )
            with gr.Column(scale=2):
                metrics = gr.Dataframe(headers=["metric", "value"], label="Results")
        gr.Markdown("### Privacy vs utility trade-off")
        trade = gr.Image(value=initial["trade"], label="trade-off", type="filepath")
        with gr.Row():
            roc = gr.Image(label="attack ROC", type="filepath")
            conf = gr.Image(label="members vs non-members confidence", type="filepath")
        with gr.Row():
            curve = gr.Image(label="training progress", type="filepath")
            aia = gr.Image(label="attribute-inference attack (cholesterol)", type="filepath")
        run_btn.click(
            _run_experiment,
            inputs=[arch, eps],
            outputs=[trade, metrics, roc, conf, curve, aia],
        )
        sweep_btn.click(_sweep_all, outputs=[trade, metrics, roc, conf, curve, aia])
        arch.change(_epsilon_update, inputs=arch, outputs=eps)
    return demo


def launch(host="0.0.0.0", port=7860, prewarm=True, quick=False):
    if prewarm:
        groups = exp.group_records()
        todo = []
        for arch, eps in exp.sweep_configs():
            base = exp.result_key(arch, eps)
            have = max((r["rep"] for r in groups.get(base, [])), default=-1) + 1
            for rep in range(have, exp.PREWARM_REPS):
                todo.append((arch, eps, rep))
        if todo:
            print(f"prewarming {len(todo)} runs...")
            for arch, eps, rep in todo:
                exp.run_experiment(arch, eps, quick=quick, rep=rep)
    demo = build_app()
    demo.queue().launch(server_name=host, server_port=port, show_error=True)
