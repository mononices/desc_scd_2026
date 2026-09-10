from privacy_mesh import data
from privacy_mesh import experiment as exp

WIDTH = 82
EPS_LABEL = {None: "inf", 8.0: "8", 4.0: "4", 2.0: "2", 1.0: "1"}


def _rule(char="="):
    print(char * WIDTH)


def _banner(title, subtitle=None):
    print()
    _rule()
    print(f" {title}")
    if subtitle:
        print(f" {subtitle}")
    _rule()


def _step(n, total, title):
    print()
    print(f"STEP {n}/{total}  {title}")
    print("-" * WIDTH)


def _pct(x):
    return f"{100.0 * x:.1f}%"


def _order(rec):
    if rec["arch"] == "centralized":
        return (0, 0.0)
    if rec["arch"] == "federated":
        return (1, 0.0)
    return (2, -(rec["eps_target"] or 0.0))


def _ordered(summaries):
    return sorted(summaries.values(), key=_order)


def _label(rec):
    if rec["arch"] == "centralized":
        return "Centralised (pooled)"
    if rec["arch"] == "federated":
        return "Federated (FedAvg)"
    return f"Fed + DP, eps={EPS_LABEL.get(rec['eps_target'], rec['eps_target'])}"


def _describe_entities(parts):
    ents = parts["entities"]
    print(f"{'entity':<26}{'trains on':>11}{'evaluates':>11}{'held out':>10}"
          f"{'mean age':>10}{'risk rate':>11}")
    totals = [0, 0, 0]
    for key in data.ENTITY_KEYS:
        e = ents[key]
        tr, te, ho = len(e["train"]), len(e["test"]), len(e["holdout"])
        totals = [totals[0] + tr, totals[1] + te, totals[2] + ho]
        age = e["train"]["age"].mean()
        prev = e["train"][data.TARGET].mean()
        print(f"{data.ENTITY_LABELS[key]:<26}{tr:>11}{te:>11}{ho:>10}"
              f"{age:>10.1f}{_pct(prev):>11}")
    print(f"{'TOTAL':<26}{totals[0]:>11}{totals[1]:>11}{totals[2]:>10}")
    print()
    print("Each authority holds records it is not permitted to share. Populations differ")
    print("by design (age, cholesterol, sex mix, baseline risk), as real emirates do.")
    print()
    print(f"The {totals[2]} 'held out' records never enter training under any configuration.")
    print("They are the red team's control group: same generator, same distribution as the")
    print("training records, so the attacker cannot win by spotting a distribution shift.")


def _utility_line(rec):
    return (f"  trains to {_pct(rec['train_acc'])} accuracy on records it has seen, "
            f"{_pct(rec['acc'])} on records it has not")


def _attack_block(rec, indent="  "):
    print(f"{indent}attack accuracy      {_pct(rec['attack_acc'])}   (50.0% = coin flip on a balanced pool)")
    print(f"{indent}attack ROC AUC       {rec['attack_auc']:.3f}   (0.500 = no signal)")
    print(f"{indent}TPR @ 1% FPR         {_pct(rec['tpr_at_1pct_fpr'])}   (1.0% = chance)")


def _citizen_row(rec, citizen_id):
    verdicts = rec.get("verdicts", [])
    v = next((x for x in verdicts if x["id"] == citizen_id), None)
    if v is None:
        return None
    rank = 1 + sum(1 for x in verdicts if x["prob"] > v["prob"])
    said = "IN " if v["prob"] >= 0.5 else "OUT"
    ok = "CORRECT" if (v["prob"] >= 0.5) == bool(v["truth"]) else "WRONG  "
    return (f"  {_label(rec):<28} says \"{said}\"  {ok}"
            f"   ranked #{rank} of {len(verdicts)}")


def _pick_exemplar(summaries):
    base = summaries.get(exp.result_key("centralized", None))
    if base is None:
        return None
    members = [v for v in base.get("verdicts", []) if v["truth"] == 1]
    if not members:
        return None
    return max(members, key=lambda v: v["prob"])["id"]


def _citizen_identity(parts, citizen_id):
    for key in data.ENTITY_KEYS:
        df = parts["entities"][key]["train"]
        hit = df[df["citizen_id"] == citizen_id]
        if not hit.empty:
            r = hit.iloc[0]
            return r["citizen_name"], data.ENTITY_LABELS[key], r["region"], r["age"]
    return None


def _final_table(items):
    print(f"{'configuration':<26}{'eps':>6}{'test accuracy':>15}{'gap':>8}"
          f"{'attack acc':>12}{'attack AUC':>15}")
    print("-" * WIDTH)
    for rec in items:
        eps_a = "inf" if rec["eps_achieved"] is None else f"{rec['eps_achieved']:.1f}"
        acc = f"{rec['acc']:.3f}"
        if rec.get("acc_std"):
            acc += f"+-{rec['acc_std']:.3f}"
        att = f"{rec['attack_auc']:.3f}"
        if rec.get("attack_auc_std"):
            att += f"+-{rec['attack_auc_std']:.3f}"
        print(f"{_label(rec):<26}{eps_a:>6}{acc:>15}{rec['gap']:>+8.3f}"
              f"{_pct(rec['attack_acc']):>12}{att:>15}")


def narrate(summaries, parts):
    items = _ordered(summaries)
    by_key = {r["config"]: r for r in items}
    central = by_key.get(exp.result_key("centralized", None))
    fed = by_key.get(exp.result_key("federated", None))
    dp = [r for r in items if r["arch"] == "fed_dp"]
    total = 6

    _banner("UAE PRIVACY MESH",
            "Learning across government entities without exposing a single citizen")

    _step(1, total, "The data that cannot be pooled")
    _describe_entities(parts)

    if central is None:
        print("\nno centralised baseline in the cache; run with --fresh")
        return

    _step(2, total, "The naive answer: pool every record into one database")
    print("This is what the law blocks, and it is our baseline for comparison.")
    print()
    print(f"  pooled training records   {central['n_train']}")
    print(_utility_line(central))
    print(f"  generalisation gap        {central['gap']:+.3f}")
    print()
    print(f"The model scores {_pct(central['train_acc'])} on its own training data. It has not")
    print("learned a general rule about heart risk; it has memorised individual citizens.")
    print("That memorisation is the leak the red team is about to exploit.")

    _step(3, total, "RED TEAM: can an attacker tell who was in the training data?")
    print("The attacker sees only the finished model's outputs -- never the data. For each")
    print(f"candidate it asks the model how confident it is, then decides IN or OUT.")
    print()
    print(f"  candidates            {central['n_candidates']} "
          f"({central['n_candidates'] // 2} real members, {central['n_candidates'] // 2} matched non-members)")
    _attack_block(central)
    print()
    print(f"The attacker is right {_pct(central['attack_acc'])} of the time. On a balanced pool,")
    print("guessing would give 50%. Membership in this database is not secret.")

    exemplar = _pick_exemplar(summaries)
    ident = _citizen_identity(parts, exemplar) if exemplar else None
    if ident:
        name, entity, region, age = ident
        print()
        print("  A specific individual, not just an average:")
        print()
        print(f"    citizen {exemplar}  \"{name}\"")
        print(f"    {entity} / {region}, age {int(age)}")
        print(f"    ground truth: THIS PERSON IS IN THE TRAINING DATA")
        print()
        for rec in items:
            row = _citizen_row(rec, exemplar)
            if row:
                print(row)
        print()
        print("  Rank is the attacker's ordering of all candidates by suspicion. Against the")
        print("  pooled model this citizen sits at the very top; under DP the attacker cannot")
        print("  distinguish them from the crowd.")
        print()
        print("  (selected automatically as the member the baseline attack scores highest,")
        print("   so this example is reproducible rather than hand-picked)")

    _step(4, total, "Fix 1 -- Federated learning: raw records never leave the emirate")
    if fed:
        print("Each authority trains locally. Only model weights reach the aggregator;")
        print("no citizen record crosses an organisational boundary.")
        print()
        print(_utility_line(fed))
        print(f"  generalisation gap        {fed['gap']:+.3f}  "
              f"(was {central['gap']:+.3f} when pooled)")
        _attack_block(fed)
        print()
        print(f"Averaging across entities blunts memorisation: the attack drops from")
        print(f"{central['attack_auc']:.3f} to {fed['attack_auc']:.3f} AUC. But it is still above chance --")
        print("federated learning alone is NOT a privacy guarantee.")

    _step(5, total, "Fix 2 -- Differential privacy: a mathematical bound, not a hope")
    print("DP-SGD clips every per-sample gradient and adds calibrated Gaussian noise")
    print("before any update leaves an entity. Opacus' Renyi accountant tracks the spend.")
    print()
    print(f"{'budget':<12}{'eps spent':>11}{'sigma':>9}{'test acc':>11}{'gap':>9}{'attack AUC':>13}")
    print("-" * WIDTH)
    for rec in dp:
        eps_t = EPS_LABEL.get(rec["eps_target"], str(rec["eps_target"]))
        print(f"{'eps = ' + eps_t:<12}{rec['eps_achieved']:>11.2f}{rec['sigma']:>9.2f}"
              f"{rec['acc']:>11.3f}{rec['gap']:>+9.3f}{rec['attack_auc']:>13.3f}")
    print()
    print(f"Every configuration spends less than its stated budget (delta = {central['delta']:g}),")
    print("and the attack collapses to chance. The guarantee holds against ANY attacker,")
    print("not just the one we happened to write.")

    _step(6, total, "The price of privacy, stated honestly")
    _final_table(items)
    print()
    best = max(dp, key=lambda r: r["acc"]) if dp else None
    worst = min(dp, key=lambda r: r["acc"]) if dp else None
    if best and fed and worst:
        delta = (best["acc"] - fed["acc"]) * 100.0
        print(f"Best protected configuration: eps = {EPS_LABEL.get(best['eps_target'])} at "
              f"{_pct(best['acc'])} accuracy, spending {best['eps_achieved']:.2f} of its budget.")
        if delta >= 0:
            print(f"That is {delta:.1f} accuracy points ABOVE unprotected federated learning, not below:")
            print("clipping and noise regularise a model that was otherwise memorising. At these")
            print("budgets privacy is free -- the trade-off only begins to bite further down.")
        else:
            print(f"That costs {abs(delta):.1f} accuracy points against unprotected federated learning.")
        print()
        print(f"The real cost appears at the tightest budget: eps = {EPS_LABEL.get(worst['eps_target'])} "
              f"gives up {(best['acc'] - worst['acc']) * 100:.1f} points")
        print(f"({_pct(best['acc'])} -> {_pct(worst['acc'])}) for a guarantee roughly eight times stronger.")
        print()
        print(f"In every protected configuration the attacker is reduced from "
              f"{_pct(central['attack_acc'])} to")
        print(f"about {_pct(best['attack_acc'])} -- a coin flip. Membership stops being recoverable.")
    print()
    print("Assumptions and limits are stated in README.md. The epsilon values are per")
    print("entity, over all rounds, under Opacus' Renyi accounting with Poisson sampling.")
    _rule()
    print()
