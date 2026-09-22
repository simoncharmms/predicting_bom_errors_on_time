# Prediction Agent — optimisation proposal for the BOM error ML setup

## 1. Where the current setup stands

The published pipeline is a **batch, stateless, single-shot** system:

```
raw_bom_data.csv → Procrustes (rho_v) → chi² → k-means (clus)
                 → IsolationForest (anom) → multi-output MLP → mlp_*_pred.csv
```

Measured on the verified run (350,263 rows, 0.24 % positives):

| configuration | accuracy | recall | precision | true positives |
| --- | --- | --- | --- | --- |
| under-trained (5 epochs) | 0.55 | 0.73 | 0.004 | 636 of 868 |
| converged / majority-class | 0.998 | 0.00 | — | 0 |

That is the core problem the agent has to solve. The model either floods the
engineer with ~157k false positives or predicts "no error" for everything. There
is no mechanism to learn which of its alerts a human actually acted on, and every
run starts from zero knowledge.

Three structural gaps, which map exactly onto the three capabilities requested:

| Gap in the current setup | Prediction Agent capability |
| --- | --- |
| `rho_v` is a single scalar per `(timestamp, part)`; error *patterns* are never named or reused | **BDQV-based pattern detection** |
| The configuration session that produced the BOM is invisible to the model | **Configuration-behaviour learning** |
| Every stage is a fresh fit; nothing is retained across vehicle generations | **Shared knowledge base** |

## 2. Agent architecture

```
                        ┌───────────────────────────────────┐
                        │      Shared Knowledge Base        │
                        │  temporal KG + pattern registry   │
                        │  + vector index over evidence     │
                        └───┬───────────────▲───────────────┘
              read/subscribe│               │write (versioned, provenance)
                        ┌───▼───────────────┴───────────────┐
   BDQV stream ────────▶│         Prediction Agent          │
   config events ──────▶│  ┌─────────────────────────────┐  │
   BOM snapshots ──────▶│  │ Perception   → BDQV encoder │  │
                        │  │ Pattern      → registry match│ │
                        │  │ Behaviour    → session model │  │
                        │  │ Forecast     → when + where  │  │
                        │  │ Explanation  → KG subgraph   │  │
                        │  └─────────────────────────────┘  │
                        └───┬───────────────┬───────────────┘
                  risk-ranked│               │hypotheses / evidence requests
                   worklist  ▼               ▼
              engineer (feedback loop)   peer agents
                                         (Config Agent, Change Agent,
                                          Supplier Agent, Release Agent)
```

### 2.1 BDQV layer — from one scalar to a typed violation vocabulary

A BDQV (BOM data quality violation) is a *typed, localised, timestamped*
assertion, not a score:

```python
BDQV = {
  "id":        "bdqv:qty_mismatch:F30:asm_4412:VS0",
  "type":      "quantity_mismatch",      # from a closed taxonomy
  "scope":     {"component": 4412, "part": 91021, "level": 3},
  "timestamp": "VS0",                    # maturity phase, not wall clock
  "severity":  0.82,
  "evidence":  {"rho_v": 1.41, "rho_v_z": 3.2, "delta_prev_phase": 0.6},
  "source":    "rule|detector|engineer",
}
```

Concrete optimisations this unlocks:

1. **Multi-label instead of one binary target.** Replace the single `erroneous`
   head with one head per BDQV type (~15-30 types). A quantity mismatch and a
   missing-variant-assignment have different signatures; collapsing them into one
   label is what forces the MLP into majority-class collapse.
2. **Keep `rho_v`, but as a feature family, not a verdict.** The repo's own
   figures state that `rho_v` "solely cannot be seen as a good indicator". So
   expose `rho_v`, its z-score within `(component, timestamp)`, its delta to the
   previous maturity phase, and its rank percentile. The Procrustes work is the
   valuable part; the single-number summary throws most of it away.
3. **Replace `IsolationForest` on a random column list with a BDQV-conditioned
   detector.** Today the detector receives the label itself (leakage, see
   `SETUP_AND_FIXES.md` §2) and a `contamination=0.05` prior against a 0.24 %
   base rate — a 20x mis-specification. Fit one detector per BDQV type with the
   contamination set from the registry's observed rate for that type.
4. **Pattern registry.** Frequent BDQV co-occurrence subgraphs (e.g.
   `late_release → quantity_mismatch → variant_gap` within two phases) are mined
   once and stored as named patterns with a hit rate and a confirmed-by-engineer
   count. At inference the agent matches against the registry first; a registry
   hit is a cheap, explainable, high-precision alert that needs no retraining.

### 2.2 Behaviour layer — learning from the configuration session

The BOM is the *output* of a configuration process, and the process leaks
information the finished BOM does not contain. Log it as events:

```
config_event(user_role, session_id, timestamp, action,
             target_node, dwell_ms, undo_count, help_opened,
             variant_switches, copy_from_generation)
```

Derived session features that plausibly predict error, ordered by expected value:

| Feature | Rationale |
| --- | --- |
| `undo_count`, `redo_count` on a subtree | hesitation marks uncertainty; uncertainty precedes error |
| `dwell_ms` z-score vs. the user's own median for that node type | unusually fast edits on complex nodes |
| `copy_from_generation` depth | carried-over structure is the documented source of the F30→G20 `rho_v` shift |
| `variant_switches` before commit | thrash in variant assignment |
| `time_since_last_phase_change` | edits made right before a phase gate |
| `role`, `tenure_bucket`, `first_time_on_this_component` | competence priors, aggregated |

Two important design constraints:

- **Personalisation without surveillance.** Model the *behaviour type*, not the
  person: features are z-scored against each user's own baseline and only
  aggregated cohorts are persisted. Nothing an engineer does becomes a
  performance metric about them. This is a hard requirement for works-council
  acceptance in a German OEM setting, and it also happens to produce a better
  model (relative deviation beats absolute speed).
- **Online, not batch.** The behaviour model is a small incrementally updated
  component (e.g. `SGDClassifier`/`partial_fit`, or a per-user Bayesian
  intercept). It adapts within a session; the heavy structural model stays batch.

### 2.3 Shared knowledge base — what the agents actually share

A **temporal knowledge graph** plus a pattern registry, with the graph as the
join key between agents:

```
(:Part)-[:PART_OF {phase}]->(:Component)-[:BELONGS_TO]->(:Generation)
(:Part)-[:HAS_VIOLATION {severity, phase}]->(:BDQV)
(:BDQV)-[:INSTANCE_OF]->(:PatternType)
(:BDQV)-[:CONFIRMED_BY {ts, verdict}]->(:Engineer)
(:ConfigSession)-[:TOUCHED {dwell, undos}]->(:Part)
(:Part)-[:SUPPLIED_BY]->(:Supplier)      ← Supplier Agent
(:Part)-[:CHANGED_BY]->(:ChangeRequest)  ← Change Agent
```

Contract rules that keep a multi-agent KB usable:

1. **Append-only with provenance.** Every write carries agent id, model version,
   confidence and timestamp. Agents may contradict each other; the graph records
   both and the consumer resolves.
2. **Read-your-writes, no cross-agent overwrites.** An agent can only retract its
   own assertions.
3. **Confidence decay.** A pattern not confirmed for N phases decays and drops
   out of the high-precision path.
4. **Shared BDQV taxonomy as the interface.** Agents exchange BDQVs, not feature
   vectors — the only way a Supplier Agent's "this supplier's parts arrive late"
   can reach the Prediction Agent without coupling their models.

Why this matters quantitatively: the F30→G20 `rho_v` shift documented in the
README currently means a model trained on one generation transfers poorly. With
a shared KG, a new generation starts with the pattern registry and the
supplier/change context of its predecessors — **warm start instead of cold
start**, which is the single largest available win on a 0.24 % positive rate.

## 3. Concrete model changes, ranked by effort vs. payoff

| # | Change | Effort | Expected effect |
| --- | --- | --- | --- |
| 1 | Fix the merge inflation and the anomaly-detector label leakage | done | current numbers were not measuring what they claimed |
| 2 | Report PR-AUC and precision@k; drop accuracy as the headline | S | makes every later change measurable |
| 3 | Split by `timestamp`, not randomly; group by `part` | S | removes optimistic bias; the task *is* temporal |
| 4 | `class_weight` / focal loss on the classification head | S | stops majority-class collapse without resampling |
| 5 | Per-BDQV-type multi-label heads | M | separates signatures that currently cancel out |
| 6 | Contamination per BDQV type from the registry | S | fixes the 5 % vs 0.24 % mis-specification |
| 7 | Session/behaviour features as a second input branch | M | adds genuinely new signal, not a re-encoding of the BOM |
| 8 | Replace k-means "contextualisation" with KG-derived context (generation, supplier, change cluster) | M | semantic context instead of a cluster id on three columns |
| 9 | GNN over the BOM adjacency in place of the flat MLP | L | the data is a hierarchy; an MLP on 10 scalars discards it |
| 10 | Human-in-the-loop: log every accept/dismiss back into the KB | M | turns precision into a learnable quantity |

Ordering note: items 2-4 are a day of work and should precede anything
architectural, because without 2 and 3 you cannot tell whether 5-10 helped.

## 4. Calibration and the alert budget

A prescriptive system is only adopted if its worklist is short. Define the
operating point explicitly rather than at the 0.5 threshold the current code
implies via `np.around`:

- Fix a **daily alert budget** k (e.g. 50 parts per component per phase).
- Optimise **precision@k** and **lead time** (phases between alert and the error
  actually materialising) — a correct alert one phase before the gate is worth far
  more than the same alert after it.
- Calibrate probabilities per BDQV type (isotonic regression) so severity is
  comparable across types when ranking the single merged worklist.
- Track **precision@k over time**: if the shared KB is working, it rises across
  vehicle generations. That is the falsifiable claim for the whole agent design.

## 5. Control models — is the agent earning its complexity?

Every added component needs an ablation that isolates its contribution:

| Model | Isolates |
| --- | --- |
| A: flat MLP on `rho_v` + features (current repo, fixed) | baseline |
| B: A + BDQV multi-label heads | typed violation vocabulary |
| C: B + session/behaviour branch | configuration behaviour |
| D: C + KG context features | shared knowledge base |
| E: D + pattern registry warm start | cross-generation transfer |
| D-shuffled | KG edges permuted — does the *semantics* contribute, or just the extra dimensions? |
| C-shuffled | session features permuted within user |

The shuffled controls are the important ones: they are the only way to show the
knowledge base contributes structure rather than capacity.

## 6. Minimal integration surface

The agent can sit on top of the repo without rewriting it:

```python
class PredictionAgent:
    def __init__(self, kb: KnowledgeBase, registry: PatternRegistry): ...

    def observe(self, bom_snapshot, config_events) -> list[BDQV]:
        """Procrustes/rho_v features + rules + detectors -> typed BDQVs."""

    def predict(self, bdqvs) -> list[Alert]:
        """Registry match (high precision) ∪ model score (recall), then rank
        to the alert budget. Each alert carries its KG evidence subgraph."""

    def explain(self, alert) -> Subgraph: ...

    def learn(self, feedback) -> None:
        """Engineer verdicts -> online behaviour update + KB write."""

    def share(self) -> list[Assertion]:
        """Publish confirmed patterns for peer agents."""
```

`observe` reuses `solve_orthogonal_procrustes`; `predict` reuses the trained
multi-output model as one scorer among several. The pipeline becomes the agent's
perception layer rather than the whole system.
