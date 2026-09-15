import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

from privacy_mesh import attack as atk
from privacy_mesh import data
from privacy_mesh import experiment as exp


@pytest.fixture(scope="module", autouse=True)
def isolated_results(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("results")
    saved = (exp.RESULTS_DIR, exp.CHART_DIR)
    exp.RESULTS_DIR = str(tmp)
    exp.CHART_DIR = str(tmp / "charts")
    yield
    exp.RESULTS_DIR, exp.CHART_DIR = saved


@pytest.fixture(scope="module")
def parts():
    return data.ensure_partitions()


def test_partitions_have_three_disjoint_splits(parts):
    for key in data.ENTITY_KEYS:
        ent = parts["entities"][key]
        ids = {s: set(ent[s]["citizen_id"]) for s in ("train", "test", "holdout")}
        assert ids["train"] and ids["test"] and ids["holdout"]
        assert not ids["train"] & ids["test"]
        assert not ids["train"] & ids["holdout"]
        assert not ids["test"] & ids["holdout"]


def test_citizen_ids_are_globally_unique(parts):
    frames = [parts["entities"][k][s]
              for k in data.ENTITY_KEYS for s in ("train", "test", "holdout")]
    frames.append(parts["entities"]["external"])
    allrows = pd.concat(frames, ignore_index=True)
    assert allrows["citizen_id"].nunique() == len(allrows)


def test_partitions_are_cached_not_regenerated(parts):
    again = data.ensure_partitions()
    a = parts["entities"]["abu_dhabi"]["train"]
    b = again["entities"]["abu_dhabi"]["train"]
    pd.testing.assert_frame_equal(a, b)
    assert np.allclose(parts["mean"], again["mean"])


def test_attack_pool_is_balanced_and_membership_is_truthful(parts):
    members, nonmembers = atk.build_candidates(parts, n_members=900, n_nonmembers=900, seed=1)
    assert len(members) == len(nonmembers) == 900
    assert not set(members["citizen_id"]) & set(nonmembers["citizen_id"])
    train_ids = set()
    holdout_ids = set()
    for key in data.ENTITY_KEYS:
        train_ids |= set(parts["entities"][key]["train"]["citizen_id"])
        holdout_ids |= set(parts["entities"][key]["holdout"]["citizen_id"])
    assert set(members["citizen_id"]) <= train_ids
    assert set(nonmembers["citizen_id"]) <= holdout_ids
    assert not set(nonmembers["citizen_id"]) & train_ids


def test_nonmembers_are_distribution_matched(parts):
    members, nonmembers = atk.build_candidates(parts, n_members=900, n_nonmembers=900, seed=1)
    for col in ("age", "chol", "thalach"):
        gap = abs(members[col].mean() - nonmembers[col].mean())
        spread = members[col].std()
        assert gap < 0.25 * spread, f"{col} differs by {gap:.2f} between members and non-members"


def test_external_pool_is_genuinely_shifted(parts):
    members, _ = atk.build_candidates(parts, n_members=900, n_nonmembers=900, seed=1)
    ext = parts["entities"]["external"]
    assert abs(members["age"].mean() - ext["age"].mean()) > 0.5


@pytest.mark.parametrize("eps", [8.0, 1.0])
def test_dp_run_stays_within_its_budget(eps):
    rec = exp.run_experiment("fed_dp", eps=eps, quick=True, rep=99)
    assert rec["eps_achieved"] is not None
    assert rec["eps_achieved"] <= eps, "spent more privacy budget than promised"
    assert rec["sigma"] > 0
    assert len(rec["eps_per_client"]) == len(data.ENTITY_KEYS)
    assert rec["eps_achieved"] == pytest.approx(max(rec["eps_per_client"]))


def test_unachievable_epsilon_fails_with_a_readable_message(parts):
    entity_train, x_test, y_test = exp._entity_matrices(parts, parts["mean"], parts["std"])
    from privacy_mesh import federation

    with pytest.raises(ValueError, match="cannot reach epsilon"):
        federation.train_federated(
            entity_train, x_test, y_test, rounds=2, local_epochs=1,
            batch=128, eps=1e-4, delta=1e-5, seed=0,
        )


def test_records_deduplicate_by_key_on_rerun():
    before = exp.load_records()
    rec = exp.run_experiment("federated", quick=True, rep=98)
    once = exp.load_records()
    exp.run_experiment("federated", quick=True, rep=98)
    twice = exp.load_records()
    assert rec["key"] in once
    assert len(twice) == len(once), "re-running the same config duplicated a record"
    assert len(once) >= len(before)


def test_attack_reports_chance_level_metrics_on_a_random_model(parts):
    from privacy_mesh.model import RiskNet

    out = atk.run_attack(RiskNet(), parts, parts["mean"], parts["std"],
                         seed=3, n_members=600, n_nonmembers=600, include_shifted=False)
    m = out["matched"]
    assert 0.40 < m["attack_auc"] < 0.60, "untrained model should leak nothing"
    assert m["n_candidates"] == 1200
    assert len(m["verdicts"]) == 1200
