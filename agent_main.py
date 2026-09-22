#### ==========================================================================
#### Prediction Agent entry point.
#### --------------------------------------------------------------------------
"""
Runs the Prediction Agent on the output of the existing pipeline.

Prerequisite: `./run.sh` at least once, so that data/bom_data_clus_anom.csv
exists (it carries rho_v, clus and anom).

    ./agent.sh                     # full run: ladder + worklist + feedback loop
    AGENT_BUDGET=0.005 ./agent.sh  # tighter alert budget
"""
import json
import os
import time

import numpy as np
import pandas as pd

from constants import (ALERT_BUDGET_FRACTION, FP_DATA, LABEL_THRESHOLD,
                       RANDOM_SEED)
from prediction_agent import KnowledgeBase, PredictionAgent
from prediction_agent.evaluate import evaluate_scores, run_ladder
from prediction_agent.behaviour import simulate_event_log
from multi_output_mlp import temporal_split


def main():
    start = time.time()
    budget = float(os.environ.get("AGENT_BUDGET", ALERT_BUDGET_FRACTION))
    kb_path = FP_DATA + "knowledge_base.sqlite"
    if os.path.exists(kb_path):
        os.remove(kb_path)          # deterministic runs; omit to accumulate

    print("=" * 70)
    print("Prediction Agent")
    print("=" * 70)

    df = pd.read_csv(FP_DATA + "bom_data_clus_anom.csv", encoding="latin-1")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].fillna(0)
    y = (df["erroneous"] > LABEL_THRESHOLD).astype(int).to_numpy()
    print(f"Loaded {len(df):,} rows | positives {y.sum():,} ({y.mean():.3%})")

    # Same temporal split as the corrected MLP, so results are comparable.
    tr, va, te, val_phases, test_phases = temporal_split(df)
    # The agent trains on train+val (it has no early-stopping split of its own
    # that needs the validation phases held out).
    fit_mask = (tr | va).to_numpy()
    df_tr, df_te = df[fit_mask].reset_index(drop=True), df[te.to_numpy()].reset_index(drop=True)
    y_tr, y_te = y[fit_mask], y[te.to_numpy()]
    print(f"Train phases -> {len(df_tr):,} rows | test phases "
          f"{test_phases.tolist()} -> {len(df_te):,} rows")

    kb = KnowledgeBase(kb_path, agent="prediction_agent", model_version="0.1.0")
    agent = PredictionAgent(kb, alert_budget_fraction=budget, seed=RANDOM_SEED)

    ### --- observe --------------------------------------------------------
    log_tr = simulate_event_log(df_tr, seed=RANDOM_SEED)
    log_te = simulate_event_log(df_te, seed=RANDOM_SEED + 1)
    obs_tr = agent.observe(df_tr, log_tr)
    obs_te = agent.observe(df_te, log_te)

    fired = (obs_tr["bdqv"].drop(columns=["bdqv_count", "bdqv_severity_max",
                                          "bdqv_severity_sum"]) > 0)
    print("\nBDQVs derived on training rows:")
    for name, count in fired.sum().sort_values(ascending=False).items():
        rate = y_tr[fired[name].to_numpy()].mean() if count else float("nan")
        print(f"  {name:<22} fires {count:>7,} times | error rate when firing "
              f"{rate:.3f} (base {y_tr.mean():.3f})")

    ### --- knowledge base -------------------------------------------------
    agent.index_graph(df_tr, obs_tr)
    patterns = agent.mine_patterns(obs_tr["signature"], y_tr)
    print(f"\nPattern registry: {len(patterns)} signatures with support >= 50")
    print(patterns.head(8).to_string(index=False))

    ### --- fit and evaluate ----------------------------------------------
    agent.fit(df_tr, y_tr, obs_tr)
    scores = agent.score(df_te, obs_te)
    agent_metrics = evaluate_scores(y_te, scores, budget)
    print("\nAgent on held-out later phases:")
    for k, v in agent_metrics.items():
        print(f"  {k}: {v}")

    ### --- ablation ladder ------------------------------------------------
    print("\nAblation ladder (test phases, budget "
          f"{budget:.1%}):")
    ladder = run_ladder(df_tr, y_tr, obs_tr, df_te, y_te, obs_te, kb,
                        budget_fraction=budget, seed=RANDOM_SEED)
    cols = ["model", "pr_auc", "roc_auc", "precision_at_k", "recall_at_k",
            "lift_at_k", "hits_at_k"]
    print(ladder[cols].to_string(index=False,
                                 float_format=lambda x: f"{x:.4f}"))

    ### --- worklist, explanation, feedback --------------------------------
    alerts = agent.predict(df_te, obs_te)
    print(f"\nWorklist: {len(alerts)} alerts at a {budget:.1%} budget")
    for a in alerts[:5]:
        print(f"  part {a.part:>8.0f} comp {a.component:>4.0f} phase "
              f"{a.phase:>4.0f} score {a.score:.3f} | {a.signature}")
    print("\nExplanation for the top alert:")
    print(json.dumps(agent.explain(alerts[0]), indent=2, default=str))

    # Simulated engineer verdicts, so the feedback path is exercised.
    # (part, component, phase) is not unique, so verdicts are keyed on row_id.
    truth = pd.Series(y_te, index=df_te.index)
    verdicts = ["confirmed" if truth.loc[a.row_id] else "dismissed"
                for a in alerts]
    agent.learn(df_te, obs_te, alerts, verdicts)
    print(f"\nFeedback folded in: {verdicts.count('confirmed')} confirmed, "
          f"{verdicts.count('dismissed')} dismissed")
    human = kb.feedback_precision()
    if len(human):
        print(human.sort_values("human_precision", ascending=False)
                   .head(5).to_string(index=False))

    shared = agent.share()
    print(f"\nShared with peer agents: {len(shared)} patterns (lift >= 1.2)")
    print("Knowledge base contents:", kb.stats())

    report = {"agent": agent_metrics, "ladder": ladder.to_dict("records"),
              "alert_budget_fraction": budget,
              "kb_stats": kb.stats(),
              "n_shared_patterns": int(len(shared))}
    with open(FP_DATA + "agent_report.json", "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    ladder.to_csv(FP_DATA + "agent_ablation.csv", index=False)
    kb.close()
    print(f"\nWrote {FP_DATA}agent_report.json and {FP_DATA}agent_ablation.csv")
    print(f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")


if __name__ == "__main__":
    main()
