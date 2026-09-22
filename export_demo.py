"""
Export a self-contained JSON snapshot of one Prediction Agent run, for the
demo dashboard in `dashboard/`.

Everything the dashboard shows comes from here, so a reviewer can regenerate it
and diff the numbers instead of trusting a screenshot:

    ./run.sh && PYTHONPATH=src .venv/bin/python export_demo.py
"""
import json
import os

import numpy as np
import pandas as pd

from prediction_agent import KnowledgeBase, PredictionAgent
from prediction_agent.bdqv import BDQV_TYPES, SIGNATURE_TYPES
from prediction_agent.behaviour import simulate_event_log, behaviour_features
from prediction_agent.evaluate import run_ladder, evaluate_scores

SRC = "data/bom_data_clus_anom.csv"
OUT = "dashboard/demo_data.json"
BUDGET = float(os.environ.get("AGENT_BUDGET", 0.01))
SEED = 1
MAX_RANK = 2400          # deepest budget the dashboard slider can reach
N_ALERTS = 40            # alerts shipped with full evidence


def main() -> None:
    df = pd.read_csv(SRC).fillna(0)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    y = (df["erroneous"] > 0).astype(int).to_numpy()

    phases = np.sort(df["timestamp"].unique())
    share = df["timestamp"].value_counts(normalize=True).sort_index().cumsum()
    cut = float(share[share >= 0.93].index[0])
    tr, te = df["timestamp"] <= cut, df["timestamp"] > cut
    df_tr, df_te = df[tr].copy(), df[te].copy()
    y_tr, y_te = y[tr.to_numpy()], y[te.to_numpy()]

    kb_path = "data/demo_kb.sqlite"
    if os.path.exists(kb_path):
        os.remove(kb_path)
    kb = KnowledgeBase(kb_path)
    agent = PredictionAgent(kb, alert_budget_fraction=BUDGET, seed=SEED)

    obs_tr = agent.observe(df_tr)
    obs_te = agent.observe(df_te)
    agent.index_graph(df_tr, obs_tr)
    patterns = agent.mine_patterns(obs_tr["signature"], y_tr)
    agent.fit(df_tr, y_tr, obs_tr)

    # --- BDQV firing behaviour on training rows ---------------------------
    bdqv_rows = []
    base_tr = float(y_tr.mean())
    for t in BDQV_TYPES:
        fires = (obs_tr["bdqv"][t] > 0).to_numpy()
        bdqv_rows.append({
            "type": t,
            "fires": int(fires.sum()),
            "rate": float(y_tr[fires].mean()) if fires.sum() else None,
            "base": base_tr,
            "in_signature": t in SIGNATURE_TYPES,
        })
    bdqv_rows.sort(key=lambda r: -r["fires"])

    # --- ablation ladder, keeping the raw score vectors -------------------
    ladder, scores = run_ladder(df_tr, y_tr, obs_tr, df_te, y_te, obs_te, kb,
                                budget_fraction=BUDGET, seed=SEED,
                                return_scores=True)

    # For any k <= MAX_RANK, precision@k is cumsum(hits)/k. Shipping the label
    # sequence in score order makes the budget slider exact rather than
    # interpolated, at a few kB per model.
    ranked = {name: [int(v) for v in y_te[np.argsort(-s)[:MAX_RANK]]]
              for name, s in scores.items()}

    # --- the worklist an engineer would actually receive ------------------
    alerts = agent.predict(df_te, obs_te)
    truth = pd.Series(y_te, index=df_te.index)
    verdicts = ["confirmed" if truth.loc[a.row_id] else "dismissed"
                for a in alerts]
    agent.learn(df_te, obs_te, alerts, verdicts)

    alert_rows = []
    for a, v in zip(alerts[:N_ALERTS], verdicts[:N_ALERTS]):
        ex = agent.explain(a)
        alert_rows.append({
            "part": a.part, "component": a.component, "phase": a.phase,
            "score": a.score, "signature": a.signature,
            "bdqv_types": list(a.bdqv_types), "verdict": v,
            "why": ex["why"],
            "evidence": {k: (float(x) if isinstance(x, (int, float, np.floating))
                             else x)
                         for k, x in ex["evidence"].items()},
            "graph_context": [list(e) for e in ex["graph_context"][:6]],
        })

    feedback = kb.feedback_precision()

    payload = {
        "meta": {
            "rows": int(len(df)), "positives": int(y.sum()),
            "base_rate": float(y.mean()),
            "train_rows": int(len(df_tr)), "test_rows": int(len(df_te)),
            "train_phases": [float(p) for p in phases if p <= cut],
            "test_phases": [float(p) for p in phases if p > cut],
            "test_base_rate": float(y_te.mean()),
            "budget": BUDGET, "max_rank": MAX_RANK, "seed": SEED,
        },
        "phase_sizes": [{"phase": float(p),
                         "rows": int((df["timestamp"] == p).sum()),
                         "rate": float(y[(df["timestamp"] == p).to_numpy()].mean())}
                        for p in phases],
        "label_truncation": {
            "distinct_values": int(df["erroneous"].nunique()),
            "max": float(df["erroneous"].max()),
            "exact_zero": int((df["erroneous"] == 0).sum()),
            "astype_int_positives": int(df["erroneous"].astype(int).sum()),
            "gt_zero_positives": int((df["erroneous"] > 0).sum()),
        },
        "bdqv": bdqv_rows,
        "signature_types": list(SIGNATURE_TYPES),
        "patterns": json.loads(
            patterns.sort_values("lift", ascending=False)
            .head(12)[["signature", "support", "hits", "precision", "lift"]]
            .to_json(orient="records")),
        "ladder": json.loads(ladder.to_json(orient="records")),
        "ranked_labels": ranked,
        "alerts": alert_rows,
        "feedback": json.loads(feedback.to_json(orient="records"))
        if len(feedback) else [],
        "kb_stats": kb.stats(),
    }

    os.makedirs("dashboard", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh)
    print(f"Wrote {OUT} ({os.path.getsize(OUT)/1024:.0f} kB)")
    print(f"models: {list(ranked)}")
    print(f"alerts: {len(alert_rows)} | patterns: {len(payload['patterns'])}")


if __name__ == "__main__":
    main()
