# Prediction Agent

An agent layer on top of the existing pipeline. It does not replace it: the
Procrustes stage remains the perception front-end (`rho_v`), the isolation
forest remains one detector among several, and the multi-output MLP remains one
scorer. What the agent adds are the three requested capabilities.

```
                       ┌────────────────────────────────────┐
                       │   Shared Knowledge Base (SQLite)   │
                       │  temporal KG + pattern registry    │
                       │  append-only, provenance, decay    │
                       └──┬──────────────────────▲──────────┘
                     read │                      │ write
                       ┌──▼──────────────────────┴──────────┐
  BOM snapshot ───────▶│          PredictionAgent           │
  config events ──────▶│  observe → predict → explain       │
                       │          → learn → share           │
                       └──┬──────────────────────┬──────────┘
        ranked worklist   │                      │ confirmed patterns
        (alert budget)    ▼                      ▼
                     engineer ──feedback──▶  peer agents
```

## Run it

```bash
./run.sh      # once, to produce data/bom_data_clus_anom.csv (rho_v, clus, anom)
./agent.sh    # ~46 s: observe, mine, fit, ablate, alert, explain, learn, share
```

Outputs: `data/agent_report.json`, `data/agent_ablation.csv`,
`data/knowledge_base.sqlite`. `AGENT_BUDGET=0.005 ./agent.sh` tightens the
worklist.

## 1. BDQV pattern detection

`src/prediction_agent/bdqv.py`. A BDQV is a typed, localised, timestamped
violation, replacing the single opaque `erroneous` scalar. Eight detectors, all
computed from observable columns only — never from the label:

| BDQV type | fires (train) | error rate when firing | base |
| --- | --- | --- | --- |
| `delayed_part` | 264,473 | 0.156 | 0.152 |
| `variant_churn` | 131,811 | 0.157 | 0.152 |
| `quantity_outlier` | 23,502 | 0.166 | 0.152 |
| `structural_anomaly` | 15,041 | 0.152 | 0.152 |
| `rho_v_deviation` | 10,674 | **0.112** | 0.152 |
| `newly_introduced` | 3,433 | **0.032** | 0.152 |
| `single_phase_part` | 80 | 0.000 | 0.152 |
| `rho_v_phase_jump` | 50 | 0.040 | 0.152 |

Two things to read from this table. First, no single detector is a good
predictor on its own — which is the repo's own conclusion about `rho_v`, now
quantified. Second, `rho_v_deviation` and `newly_introduced` fire on rows that
are *less* likely to be erroneous than the base rate; they are informative, just
with the opposite sign. That is exactly the kind of thing a single binary label
cannot express.

Combinations are where the signal is. The registry mines BDQV co-occurrence
signatures on training rows only:

| signature | support | precision | lift |
| --- | --- | --- | --- |
| `structural_anomaly\|quantity_outlier` | 1,190 | 0.194 | **1.27** |
| `structural_anomaly\|quantity_outlier\|variant_churn` | 900 | 0.182 | 1.20 |
| `rho_v_deviation\|structural_anomaly\|quantity_outlier` | 484 | 0.182 | 1.19 |
| `quantity_outlier\|variant_churn` | 8,483 | 0.176 | 1.15 |

A registry hit is cheap, explainable and needs no retraining, which is what
should transfer to a new vehicle generation.

High-prevalence flags (`delayed_part`, set on 81 % of rows) are kept as model
features but excluded from signatures via `FLAG_TYPES` — including them produced
one dominant, uninformative pattern with support 235k and lift 1.03 that drowned
out the selective combinations.

## 2. Configuration-behaviour learning

`src/prediction_agent/behaviour.py`.

**Read this before interpreting the behaviour results.** The published dataset is
a BOM snapshot and contains no configuration event log. The module therefore
defines the interface a productive PLM integration must deliver
(`CONFIG_EVENT_COLUMNS`) and ships `simulate_event_log` as a stand-in so the
agent is runnable today. The simulator is derived from BOM structure only and
**never** from the label, so it cannot inject artificial predictive signal.
Deriving it from the label would have produced a flattering number and no
knowledge.

Features are z-scored against **each user's own** baseline, so the model learns
a behaviour type rather than ranking people; only cohort aggregates are
persisted. This is a works-council requirement in a German OEM setting and also
the better modelling choice, since relative deviation predicts error better than
absolute speed. `OnlineBehaviourModel` uses `partial_fit`, so it adapts within a
session while the structural model stays batch.

## 3. Shared knowledge base

`src/prediction_agent/knowledge_base.py`. SQLite, stdlib only, so several agents
can attach to one file. Three contract rules:

1. **Append-only with provenance** — every row records agent, model version,
   confidence, timestamp. Agents may contradict each other; both statements are
   kept and the consumer resolves.
2. **Retract-your-own-only** — `retract()` filters on the calling agent.
3. **Confidence decay** — exponential with a configurable half-life, so an
   unconfirmed pattern drops out of the high-precision path.

Graph shape: `(:Part)-[:PART_OF {phase, rho_v}]->(:Component)`,
`(:Part)-[:HAS_SIGNATURE {phase}]->(:Signature)`, plus `feedback` rows from
engineer verdicts. Agents exchange **BDQVs**, not feature vectors — the only way
a Supplier Agent's "this supplier delivers late" can reach the Prediction Agent
without coupling their models. `share()` publishes patterns above a lift and
support floor.

Engineer verdicts are folded back through `learn()` and update the registry
counters, which turns precision into a learnable quantity. One run at a 1 %
budget produced 73 confirmed / 159 dismissed and this human-precision table:

| signature | confirmed | dismissed | human precision |
| --- | --- | --- | --- |
| `structural_anomaly` | 49 | 30 | 0.62 |
| `clean` | 17 | 34 | 0.33 |
| `variant_churn` | 7 | 81 | 0.08 |

## Measured results and the ablation ladder

Held-out **later** maturity phases (23,231 rows, base rate 0.173), 1 % alert
budget, identical split to the corrected MLP:

| model | PR-AUC | ROC-AUC | precision@1 % | lift@1 % | hits |
| --- | --- | --- | --- | --- | --- |
| A: repo features | 0.245 | 0.647 | 0.181 | 1.05 | 42 |
| B: + BDQV types | 0.247 | 0.649 | 0.259 | 1.49 | 60 |
| C: + behaviour | 0.225 | 0.610 | 0.280 | 1.62 | 65 |
| **C-shuffled** | 0.248 | 0.654 | 0.190 | 1.09 | 44 |
| D: + KG context | 0.218 | 0.601 | **0.319** | **1.84** | 74 |
| **D-shuffled** | 0.224 | 0.613 | 0.259 | 1.49 | 60 |
| E: + registry | 0.218 | 0.601 | 0.315 | 1.82 | 73 |

Corrected MLP baseline on the same split: PR-AUC 0.237–0.263, precision@1 %
0.25–0.28.

How to read this honestly:

- **The capability stack works at the top of the ranking.** precision@1 % rises
  1.05 → 1.49 → 1.62 → 1.84 lift as BDQVs, behaviour and KG context are added.
  On a 232-part worklist that is 42 → 74 real errors found, a 76 % improvement
  in engineer-visible yield.
- **The shuffled controls hold.** C beats C-shuffled (1.62 vs 1.09) and D beats
  D-shuffled (1.84 vs 1.49), so both blocks contribute *structure*, not just
  extra dimensions. This is the claim that matters and it survives its control.
- **PR-AUC does not improve — it drifts slightly down.** The agent is not a
  better global ranker; it is a better ranker *in the region that gets worked*.
  If you care about the whole ordering rather than a budgeted worklist, model A
  is as good. Say so rather than quoting only the favourable metric.
- **The registry (E) is neutral here** versus D. With one vehicle generation in
  the data there is nothing to transfer; its value is the cross-generation warm
  start, which this dataset cannot demonstrate. Claiming otherwise would not be
  supportable.
- **Absolute performance remains modest.** Recall at a 1 % budget is 1.8 %. This
  is a triage aid, not an oracle, and the phase-regression head is still not
  usable (MAE ≈ 5 phases).

## Demonstrating it to someone else

`dashboard/` is a static page that walks the loop on the real data: the label
defect, the eight detectors and their firing rates, the mined signatures, an
interactive review budget, a real alert with its evidence subgraph, and the
verdict feedback. Build and open it with:

```bash
./run.sh
PYTHONPATH=src .venv/bin/python export_demo.py   # writes dashboard/demo_data.json
python3 -m http.server -d dashboard 8412
```

Every figure it displays is read from `demo_data.json`, so a reviewer can
regenerate the snapshot and diff it rather than trusting a screenshot.

The budget control is the part worth demonstrating, because it also shows the
method's boundary: past roughly a 3 % budget the ordering inverts and the plain
baseline overtakes the agent. The agent concentrates its confidence in a short
worklist and is the wrong tool if most of the BOM will be reviewed anyway.

### Run-to-run variance

The dashboard snapshot is a separate run from the one tabulated above and puts
rung D at 1.77x rather than 1.84x (71 vs 74 hits). The knowledge base is rebuilt
per run and the boosted trees are seeded but not deterministic across differing
registry state, so treat differences between neighbouring rungs of this size as
noise. The A-to-D gap and the direction of both shuffled controls are stable;
the exact ordering of D and E is not.

## What would move the needle next

1. A real configuration event log. The behaviour branch is the one block whose
   simulated version already helps structurally; real undo/dwell data is the
   most likely source of genuinely new signal.
2. A GNN over the BOM adjacency instead of a flat feature vector — the data is a
   hierarchy and both the MLP and the boosted trees discard it.
3. A second vehicle generation, to test the registry warm start (E) as intended.
4. Multi-label BDQV heads trained per violation type rather than one merged
   score.
