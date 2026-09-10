import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

from privacy_mesh import experiment as exp


def _utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_sweep(quick=False, force=False, progress=None):
    groups = exp.group_records()
    todo = []
    for arch, eps in exp.sweep_configs():
        base = exp.result_key(arch, eps)
        if force:
            todo += [(arch, eps, r) for r in range(exp.PREWARM_REPS)]
            continue
        have = max((r["rep"] for r in groups.get(base, [])), default=-1) + 1
        todo += [(arch, eps, r) for r in range(have, exp.PREWARM_REPS)]
    if not todo:
        print("all experiment repeats already cached")
    else:
        print(f"running {len(todo)} experiment runs")
        total = len(todo)
        for i, (arch, eps, rep) in enumerate(todo):

            def cb(frac, msg, arch=arch, eps=eps, rep=rep, i=i):
                if progress:
                    progress((i + frac) / total, msg)
                print(f"[{arch} eps={eps} run {rep + 1}] {msg}")

            exp.run_experiment(arch, eps, quick=quick, progress=cb, rep=rep)
    return exp.config_summaries()


def render_charts(summaries):
    from privacy_mesh import charts

    if not summaries:
        return []
    central = summaries.get(exp.result_key("centralized", None))
    written = [charts.save_figure(charts.tradeoff_figure(summaries), "tradeoff.png")]
    for key, rec in summaries.items():
        written.append(charts.save_figure(charts.roc_figure(rec, reference=central), f"roc_{key}.png"))
        written.append(charts.save_figure(charts.confidence_figure(rec, reference=central), f"conf_{key}.png"))
        written.append(charts.save_figure(charts.curve_figure(rec), f"curve_{key}.png"))
    print(f"wrote {len(written)} charts to {exp.CHART_DIR}")
    return written


def print_summary(summaries):
    order = {"centralized": 0, "federated": 1}
    recs = sorted(summaries.values(), key=lambda r: (order.get(r["arch"], 2), r["eps_target"] or 0))
    print()
    hdr = f"{'architecture':<26}{'ε target':>9}{'ε achieved':>11}{'acc (±std)':>15}{'attack AUC (±std)':>20}"
    print(hdr)
    for r in recs:
        eps_t = "∞" if r["eps_target"] is None else f"{r['eps_target']:g}"
        eps_a = "∞" if r["eps_achieved"] is None else f"{r['eps_achieved']:.2f}"
        acc = f"{r['acc']:.3f} (±{r['acc_std']:.3f})"
        att = f"{r['attack_auc']:.3f} (±{r['attack_auc_std']:.3f})"
        print(f"{r['arch_label']:<26}{eps_t:>9}{eps_a:>11}{acc:>15}{att:>20}")
    print()


def run_walkthrough(fresh=False, quick=False):
    from privacy_mesh import data
    from privacy_mesh import walkthrough as wt

    if fresh:
        run_sweep(quick=quick, force=True)
    summaries = exp.config_summaries()
    if not summaries:
        print("no cached results found.")
        print("run one of:")
        print("  python -m privacy_mesh.cli walkthrough --fresh          (full, ~10 min)")
        print("  python -m privacy_mesh.cli walkthrough --fresh --quick  (reduced, ~2 min)")
        return 1
    wt.narrate(summaries, data.ensure_partitions())
    return 0


def main():
    parser = argparse.ArgumentParser(prog="privacy-mesh")
    sub = parser.add_subparsers(dest="cmd")
    pw = sub.add_parser("prewarm", help="run all experiments (3 repeats each) and cache results")
    pw.add_argument("--quick", action="store_true")
    pw.add_argument("--force", action="store_true")
    wk = sub.add_parser("walkthrough", help="narrate the whole demonstration end to end")
    wk.add_argument("--fresh", action="store_true", help="recompute instead of using the cache")
    wk.add_argument("--quick", action="store_true", help="with --fresh, use reduced settings")
    sv = sub.add_parser("serve", help="launch the gradio demo")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=7860)
    sv.add_argument("--quick", action="store_true")
    sv.add_argument("--no-prewarm", action="store_true")
    args = parser.parse_args()
    _utf8()
    try:
        return _dispatch(parser, args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nerror: {exc}")
        return 1


def _dispatch(parser, args):
    if args.cmd == "prewarm":
        summaries = run_sweep(quick=args.quick, force=args.force)
        render_charts(summaries)
        print_summary(summaries)
    elif args.cmd == "walkthrough":
        return run_walkthrough(fresh=args.fresh, quick=args.quick)
    elif args.cmd == "serve":
        if not args.no_prewarm:
            print_summary(run_sweep(quick=args.quick))
        from privacy_mesh.demo import launch

        launch(host=args.host, port=args.port)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
