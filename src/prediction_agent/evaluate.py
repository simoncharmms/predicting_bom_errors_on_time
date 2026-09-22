"""
Ablation ladder for the Prediction Agent.

Each capability must earn its complexity. The ladder isolates one block at a
time, and the two shuffled controls are the important rows: if permuting a
block leaves performance unchanged, that block contributed model capacity, not
structure.

    A   repo features only (fixed baseline)
    B   A + typed BDQV features
    C   B + configuration-behaviour features
    D   C + knowledge-graph context
    E   D + pattern-registry blending
    C-shuffled / D-shuffled   permuted controls

All models share the same temporal split, the same alert budget and the same
metrics as the corrected MLP, so the numbers are directly comparable.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .agent import BASE_FEATURES, PredictionAgent
from .bdqv import BDQV_TYPES


def precision_recall_at_k(y_true, scores, k):
    k = int(min(k, len(scores)))
    if k == 0:
        return float("nan"), float("nan"), 0
    order = np.argsort(-np.asarray(scores))[:k]
    y = np.asarray(y_true)
    hits = int(y[order].sum())
    total = int(y.sum())
    return hits / k, (hits / total if total else float("nan")), hits


def evaluate_scores(y_true, scores, budget_fraction) -> dict:
    y = np.asarray(y_true)
    k = max(1, int(round(len(scores) * budget_fraction)))
    prec_k, rec_k, hits = precision_recall_at_k(y, scores, k)
    base = float(y.mean())
    return {
        "n": int(len(scores)),
        "base_rate": base,
        "pr_auc": float(average_precision_score(y, scores)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else None,
        "k": k,
        "precision_at_k": float(prec_k),
        "recall_at_k": float(rec_k),
        "hits_at_k": hits,
        "lift_at_k": float(prec_k / base) if base > 0 else None,
    }


def _plain_model(seed):
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=seed)


def run_ladder(df_tr, y_tr, obs_tr, df_te, y_te, obs_te, kb,
               budget_fraction=0.01, seed=1) -> pd.DataFrame:
    rows = []

    # --- A: repo features only, no agent machinery -----------------------
    model = _plain_model(seed).fit(df_tr[BASE_FEATURES].astype(float), y_tr)
    scores = model.predict_proba(df_te[BASE_FEATURES].astype(float))[:, 1]
    rows.append({"model": "A: repo features",
                 **evaluate_scores(y_te, scores, budget_fraction)})

    # --- B: + typed BDQVs -------------------------------------------------
    bdqv_cols = list(BDQV_TYPES) + ["bdqv_count", "bdqv_severity_max",
                                    "bdqv_severity_sum"]
    Xtr = pd.concat([df_tr[BASE_FEATURES].astype(float), obs_tr["bdqv"][bdqv_cols]], axis=1)
    Xte = pd.concat([df_te[BASE_FEATURES].astype(float), obs_te["bdqv"][bdqv_cols]], axis=1)
    model = _plain_model(seed).fit(Xtr, y_tr)
    rows.append({"model": "B: + BDQV types",
                 **evaluate_scores(y_te, model.predict_proba(Xte)[:, 1],
                                   budget_fraction)})

    # --- C / D / E and their shuffled controls ---------------------------
    variants = [
        ("C: + behaviour",   dict(use_behaviour=True,  use_kg=False, use_registry=False), None),
        ("C-shuffled",       dict(use_behaviour=True,  use_kg=False, use_registry=False), "behaviour"),
        ("D: + KG context",  dict(use_behaviour=True,  use_kg=True,  use_registry=False), None),
        ("D-shuffled",       dict(use_behaviour=True,  use_kg=True,  use_registry=False), "kg"),
        ("E: + registry",    dict(use_behaviour=True,  use_kg=True,  use_registry=True),  None),
    ]
    for name, flags, shuffle in variants:
        agent = PredictionAgent(kb, alert_budget_fraction=budget_fraction,
                                seed=seed, **flags)
        agent.fit(df_tr, y_tr, obs_tr, shuffle=shuffle)
        scores = agent.score(df_te, obs_te, shuffle=shuffle)
        rows.append({"model": name,
                     **evaluate_scores(y_te, scores, budget_fraction)})

    return pd.DataFrame(rows)
