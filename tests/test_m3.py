"""M3 unit tests (synthetic data only): qualifier lexicon, feature builder, ranker plumbing."""
import numpy as np
import pandas as pd

from termmap.rerank.cross_encoder import CandidateText
from termmap.rerank.features import FeatureBuilder, qualifier_features, qualifiers
from termmap.rerank.ltr import fit, predict, split_groups, top1_from_scores


def test_qualifiers_and_mismatch_counts():
    q = qualifiers("acute on chronic respiratory failure with hypoxia, left, stage 3")
    assert {("acuity", "acute"), ("acuity", "chronic"), ("laterality", "left"),
            ("modifier", "hypoxia"), ("number", "stage_3"), ("relation", "with")} <= q
    f = qualifier_features(qualifiers("left knee pain"), qualifiers("pain in right knee"))
    assert f["q_laterality_conflict"] == 1 and f["code_extra"] == 1 and f["term_extra"] == 1
    assert qualifiers("ckd stage iii") == {("number", "stage_3")}


def _icd():
    table = pd.DataFrame({"code": ["I50.9", "I50.1", "I10"],
                          "text": ["heart failure, unspecified", "left ventricular failure", "essential hypertension"]})
    meta = pd.DataFrame({"code": ["I50.9", "I50.1", "I10"], "title": ["Heart failure, unspecified",
                         "Left ventricular failure", "Essential hypertension"], "category_title": ["Heart failure"] * 2 + ["Hypertension"],
                         "is_leaf": [True, True, True], "depth": [1, 1, 0]}).set_index("code")
    return table, meta


def test_feature_builder_uses_training_stats_only():
    table, meta = _icd()
    fb = FeatureBuilder("ICD10CM", table, meta, CandidateText("ICD10CM", table, meta))
    rows = pd.DataFrame({"term_norm": ["heart failure, unspecified"], "code": ["I50.9"]}, index=[5])
    train = pd.DataFrame({"term_norm": ["chf"], "display_norm": ["heart failure, unspecified"], "code": ["I50.9"]})
    cands = pd.DataFrame({"row": [5, 5], "code": ["I50.9", "I50.1"], "gold": [1, 0], "fused_rank": [1, 2], "rrf": [.03, .02],
                          "rank_char": [1, 2], "rank_dense": [2, 1], "rank_memory": [1, 999], "expanded": [0, 0],
                          "embedded": [0, 0], "ce_ctx": [2.0, np.nan]})
    f = fb.build(cands, rows, train, "ce_ctx")
    assert f["exact_dict"].tolist() == [1, 0] and f["exact_memory"].tolist() == [1, 0]
    assert f["unspecified"].tolist() == [1, 0] and f["ce_missing"].tolist() == [0, 1]
    assert f["q_laterality_code_only"].tolist() == [0, 1] and f["label"].tolist() == [1, 0]


def test_ranker_learns_synthetic_signal_and_is_deterministic():
    rng = np.random.default_rng(0)
    n_rows, k = 120, 5
    df = pd.DataFrame({"row": np.repeat(np.arange(n_rows), k), "code": np.tile([f"C{i}" for i in range(k)], n_rows)})
    df["label"] = 0
    gold = rng.integers(0, k, n_rows)
    df.loc[df.index[np.arange(n_rows) * k + gold], "label"] = 1
    df["signal"] = df["label"] + rng.normal(0, 0.3, len(df))
    df["chapter"] = 0
    cfg = {"seed": 42, "num_threads": 2}
    keep, val = split_groups(df["row"].values, 0.2, 42)
    b1, _ = fit(df[df.row.isin(keep)], cfg, ["chapter"], val=df[df.row.isin(val)])
    b2, _ = fit(df[df.row.isin(keep)], cfg, ["chapter"], val=df[df.row.isin(val)])
    s1, s2 = predict(b1, df), predict(b2, df)
    assert np.array_equal(s1, s2)
    assert top1_from_scores(df, s1, n_rows) > 0.9
