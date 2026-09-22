"""
The Prediction Agent.

observe -> predict -> explain -> learn -> share

It sits on top of the existing pipeline rather than replacing it: the Procrustes
stage stays the perception front-end (`rho_v`), the isolation forest stays one
detector among several, and the multi-output MLP becomes one scorer whose output
the agent can consume. What the agent adds is the three requested capabilities:

  * typed BDQV error patterns instead of one opaque binary label,
  * a model of user behaviour during configuration,
  * a shared knowledge base it reads from and writes to, so a new vehicle
    generation starts warm.

Alerts are always ranked and truncated to an explicit budget, because a
prescriptive system nobody can work through is not prescriptive.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .bdqv import BDQV_TYPES, FLAG_TYPES, bdqv_signature, bdqvs_for_rows, derive_bdqv_frame
from .behaviour import (BEHAVIOUR_FEATURES, OnlineBehaviourModel,
                        behaviour_features, simulate_event_log)
from .knowledge_base import Assertion, KnowledgeBase

BASE_FEATURES = ["component", "part", "feature_1", "feature_2", "feature_3",
                 "feature_4", "feature_5", "rho_v", "clus", "anom"]
KG_FEATURES = ["kg_component_risk", "kg_part_risk", "kg_component_volume",
               "kg_part_phase_span", "kg_pattern_precision", "kg_pattern_support"]


@dataclass
class Alert:
    # Stable handle on the originating row. (part, component, phase) is NOT
    # unique in a multi-level BOM, so feedback has to be keyed on the row.
    row_id: Any
    part: float
    component: float
    phase: float
    score: float
    signature: str
    bdqv_types: list[str]
    lead_phases: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


class PredictionAgent:
    """
    Parameters
    ----------
    kb : KnowledgeBase
    alert_budget_fraction : share of rows that may be emitted as alerts.
    use_behaviour / use_kg / use_registry : ablation switches, so each
        capability's contribution can be isolated (see evaluate.py).
    """

    def __init__(self, kb: KnowledgeBase, alert_budget_fraction: float = 0.01,
                 use_behaviour: bool = True, use_kg: bool = True,
                 use_registry: bool = True, seed: int = 1):
        self.kb = kb
        self.alert_budget_fraction = alert_budget_fraction
        self.use_behaviour = use_behaviour
        self.use_kg = use_kg
        self.use_registry = use_registry
        self.seed = seed
        self.model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=seed)
        self.behaviour_model = OnlineBehaviourModel(seed=seed)
        self.feature_names: list[str] = []
        self._kg_maps: dict[str, Any] = {}
        self._train_base_rate = 0.0

    # -- observe -----------------------------------------------------------
    def observe(self, df: pd.DataFrame, event_log: pd.DataFrame | None = None
                ) -> dict[str, pd.DataFrame]:
        """Turn a BOM snapshot (+ optional config events) into typed BDQVs."""
        bdqv = derive_bdqv_frame(df)
        signature = bdqv_signature(bdqv)
        if event_log is None:
            event_log = simulate_event_log(df, seed=self.seed)
        behaviour = behaviour_features(event_log, df.index)
        return {"bdqv": bdqv, "signature": signature, "behaviour": behaviour}

    # -- knowledge base: write the graph and mine the registry -------------
    def index_graph(self, df: pd.DataFrame, observed: dict, sample: int = 2000
                    ) -> None:
        """
        Write a representative slice of the BOM into the KG.

        A slice, not all 350k rows: the KG is the shared reasoning surface
        between agents, not a second copy of the warehouse.
        """
        rows = df.sample(n=min(sample, len(df)), random_state=self.seed)
        nodes = [(f"part:{r.part:g}", "Part", {"part": float(r.part)})
                 for r in rows.itertuples()]
        nodes += [(f"component:{c:g}", "Component", {"component": float(c)})
                  for c in rows["component"].unique()]
        self.kb.upsert_nodes(nodes)
        self.kb.add_edges([
            (f"part:{r.part:g}", "PART_OF", f"component:{r.component:g}",
             float(r.timestamp), {"rho_v": float(r.rho_v)})
            for r in rows.itertuples()])
        sig = observed["signature"].loc[rows.index]
        self.kb.add_edges([
            (f"part:{p:g}", "HAS_SIGNATURE", f"signature:{s}", float(t), {})
            for p, t, s in zip(rows["part"], rows["timestamp"], sig)
            if s != "clean"])

    def mine_patterns(self, signature: pd.Series, y: np.ndarray,
                      min_support: int = 50) -> pd.DataFrame:
        """
        Build the pattern registry from TRAINING rows only.

        A pattern is a BDQV co-occurrence signature with its observed hit rate.
        A registry hit is a cheap, explainable, high-precision alert that needs
        no retraining - which is exactly what should transfer to a new
        generation.
        """
        frame = pd.DataFrame({"signature": signature.to_numpy(),
                              "y": np.asarray(y)})
        agg = (frame.groupby("signature")["y"]
                    .agg(support="size", hits="sum").reset_index())
        agg = agg[agg["support"] >= min_support].copy()
        base = float(frame["y"].mean()) or 1e-9
        agg["precision"] = agg["hits"] / agg["support"]
        agg["lift"] = agg["precision"] / base
        agg["generation"] = "all"
        agg = agg.sort_values("lift", ascending=False)
        self.kb.put_patterns(agg)
        self.kb.assert_many([
            Assertion(subject=f"signature:{r.signature}",
                      predicate="predicts_error_with_precision",
                      object=f"{r.precision:.4f}",
                      confidence=min(1.0, r.support / 1000.0))
            for _, r in agg.iterrows()])
        return agg

    # -- feature assembly --------------------------------------------------
    def _fit_kg_maps(self, df: pd.DataFrame, y: np.ndarray,
                     patterns: pd.DataFrame) -> None:
        """
        Historical context, computed on TRAINING rows only and then applied
        unchanged to validation/test. This is the "warm start" the shared KB
        provides; computing it on all rows would leak the future.
        """
        frame = df.assign(_y=y)
        base = float(frame["_y"].mean())
        self._train_base_rate = base

        def smoothed(key: str, m: float) -> pd.Series:
            """
            m-estimate smoothing towards the base rate.

            `part` has very high cardinality, so a raw group mean is almost a
            per-row lookup of the label: unsmoothed it made the KG block
            actively harmful (test precision@1% fell from 0.34 to 0.10, and the
            permuted control beat it).
            """
            g = frame.groupby(key)["_y"].agg(["sum", "size"])
            return (g["sum"] + m * base) / (g["size"] + m)

        self._kg_maps = {
            "component_risk": smoothed("component", 20.0),
            "part_risk": smoothed("part", 200.0),
            "component_volume": frame.groupby("component").size().astype(float),
            "part_phase_span": frame.groupby("part")["timestamp"].nunique().astype(float),
            "pattern_precision": (
                (patterns.set_index("signature")["hits"] + 50.0 * base)
                / (patterns.set_index("signature")["support"] + 50.0)),
            "pattern_support": patterns.set_index("signature")["support"].astype(float),
        }

    def _kg_frame(self, df: pd.DataFrame, signature: pd.Series) -> pd.DataFrame:
        m = self._kg_maps
        out = pd.DataFrame(index=df.index)
        out["kg_component_risk"] = df["component"].map(
            m["component_risk"]).fillna(self._train_base_rate)
        out["kg_part_risk"] = df["part"].map(
            m["part_risk"]).fillna(self._train_base_rate)
        out["kg_component_volume"] = df["component"].map(
            m["component_volume"]).fillna(0.0)
        out["kg_part_phase_span"] = df["part"].map(
            m["part_phase_span"]).fillna(0.0)
        out["kg_pattern_precision"] = signature.map(
            m["pattern_precision"]).fillna(self._train_base_rate).to_numpy()
        out["kg_pattern_support"] = signature.map(
            m["pattern_support"]).fillna(0.0).to_numpy()
        return out[KG_FEATURES].astype(float)

    def build_matrix(self, df: pd.DataFrame, observed: dict,
                     shuffle: str | None = None,
                     rng: np.random.Generator | None = None) -> pd.DataFrame:
        """
        Assemble the design matrix from the enabled capabilities.

        `shuffle` permutes one block ("behaviour" or "kg") to provide the
        control models: if performance is unchanged, that block contributed
        capacity rather than structure.
        """
        rng = rng or np.random.default_rng(self.seed)
        blocks = [df[BASE_FEATURES].astype(float)]
        blocks.append(observed["bdqv"][list(BDQV_TYPES) +
                                      ["bdqv_count", "bdqv_severity_max",
                                       "bdqv_severity_sum"]])
        if self.use_behaviour:
            beh = observed["behaviour"].copy()
            if shuffle == "behaviour":
                beh = beh.iloc[rng.permutation(len(beh))].set_index(beh.index)
            blocks.append(beh)
        if self.use_kg:
            kg = self._kg_frame(df, observed["signature"])
            if shuffle == "kg":
                kg = kg.iloc[rng.permutation(len(kg))].set_index(kg.index)
            blocks.append(kg)
        X = pd.concat(blocks, axis=1)
        self.feature_names = list(X.columns)
        return X

    # -- fit / predict -----------------------------------------------------
    def fit(self, df: pd.DataFrame, y: np.ndarray, observed: dict,
            shuffle: str | None = None) -> "PredictionAgent":
        patterns = self.mine_patterns(observed["signature"], y)
        self._fit_kg_maps(df, y, patterns)
        X = self.build_matrix(df, observed, shuffle=shuffle)
        self.model.fit(X, y)
        if self.use_behaviour:
            self.behaviour_model.fit(observed["behaviour"], y)
        return self

    def score(self, df: pd.DataFrame, observed: dict,
              shuffle: str | None = None) -> np.ndarray:
        X = self.build_matrix(df, observed, shuffle=shuffle)
        score = self.model.predict_proba(X)[:, 1]
        if self.use_registry:
            # High-precision path: blend in the registry's observed precision
            # for this signature. Needs no retraining, transfers across
            # generations, and is fully explainable.
            reg = observed["signature"].map(
                self._kg_maps["pattern_precision"]).fillna(
                    self._train_base_rate).to_numpy()
            score = 0.8 * score + 0.2 * reg
        return score

    def predict(self, df: pd.DataFrame, observed: dict,
                budget_fraction: float | None = None) -> list[Alert]:
        """Return the ranked worklist, truncated to the alert budget."""
        budget = budget_fraction or self.alert_budget_fraction
        scores = self.score(df, observed)
        k = max(1, int(round(len(df) * budget)))
        top = np.argsort(-scores)[:k]
        idx = df.index[top]
        sig = observed["signature"].loc[idx]
        sev = observed["bdqv"].loc[idx, list(BDQV_TYPES)]
        alerts = []
        for rank, i in enumerate(idx):
            active = [t for t in BDQV_TYPES if sev.loc[i, t] > 0]
            row = df.loc[i]
            alerts.append(Alert(
                row_id=i,
                part=float(row["part"]), component=float(row["component"]),
                phase=float(row["timestamp"]), score=float(scores[top[rank]]),
                signature=str(sig.loc[i]), bdqv_types=active,
                evidence=self._evidence(i, row, str(sig.loc[i]), observed)))
        return alerts

    # -- explain -----------------------------------------------------------
    def _evidence(self, i, row, signature: str, observed: dict) -> dict:
        """
        The drivers behind one score, so the engineer can disagree with the
        agent for a stated reason rather than on instinct.
        """
        bd = observed["bdqv"].loc[i]
        ev = {
            "registry_precision": float(
                self._kg_maps["pattern_precision"].get(
                    signature, self._train_base_rate)),
            "bdqv_count": float(bd["bdqv_count"]),
            "bdqv_severity_max": float(bd["bdqv_severity_max"]),
            "rho_v": float(row["rho_v"]),
            "isolation_forest": ("anomaly" if float(row.get("anom", 0)) < 0
                                 else "normal"),
        }
        if self.use_kg:
            ev["kg_component_risk"] = float(
                self._kg_maps["component_risk"].get(
                    row["component"], self._train_base_rate))
            ev["kg_part_risk"] = float(
                self._kg_maps["part_risk"].get(
                    row["part"], self._train_base_rate))
        if self.use_behaviour and "behaviour" in observed:
            beh = observed["behaviour"].loc[i]
            for col in ("beh_session_load", "beh_dwell_z", "beh_undo_rate"):
                if col in beh.index:
                    ev[col] = float(beh[col])
        return ev

    def explain(self, alert: Alert) -> dict:
        """The evidence subgraph an engineer sees next to the alert."""
        edges = self.kb.neighbours(f"part:{alert.part:g}")
        # Flags are deliberately excluded from signatures, so an alert can show
        # "signature: clean" while a flag is firing. Say so, otherwise the
        # explanation looks self-contradictory to the engineer reading it.
        sig_hits = [t for t in alert.bdqv_types if t not in FLAG_TYPES]
        flag_hits = [t for t in alert.bdqv_types if t in FLAG_TYPES]
        n = len(sig_hits)
        return {
            "alert": f"part {alert.part:g} / component {alert.component:g} "
                     f"at phase {alert.phase:g}",
            "why": (f"BDQV signature '{alert.signature}' has historical "
                    f"precision {alert.evidence['registry_precision']:.3f} "
                    f"({n} violation{'' if n == 1 else 's'} firing: "
                    f"{', '.join(sig_hits) or 'none'})"
                    + (f"; flags set but excluded from the signature: "
                       f"{', '.join(flag_hits)}" if flag_hits else "")),
            "evidence": alert.evidence,
            "graph_context": edges[:10],
        }

    # -- learn -------------------------------------------------------------
    def learn(self, df: pd.DataFrame, observed: dict, alerts: list[Alert],
              verdicts: list[str]) -> None:
        """
        Fold engineer verdicts back in: update the online behaviour model and
        the registry counters. This is what turns precision into a learnable
        quantity instead of a number in a paper.
        """
        rows = [(a.part, a.component, a.phase, v, a.signature)
                for a, v in zip(alerts, verdicts)]
        self.kb.record_feedback(rows)
        if not self.use_behaviour or not alerts:
            return
        ids = [a.row_id for a in alerts]
        y = np.array([1 if v == "confirmed" else 0 for v in verdicts])
        # Online update on exactly the rows the engineer judged.
        self.behaviour_model.update(observed["behaviour"].loc[ids], y)

    # -- share -------------------------------------------------------------
    def share(self, min_lift: float = 1.2, min_support: int = 100
              ) -> pd.DataFrame:
        """Publish confirmed high-lift patterns for peer agents to consume."""
        patterns = self.kb.get_patterns(min_support=min_support)
        if not len(patterns):
            return patterns
        shared = patterns[patterns["lift"] >= min_lift].copy()
        self.kb.assert_many([
            Assertion(subject=f"signature:{r.signature}",
                      predicate="shared_pattern",
                      object=f"lift={r.lift:.3f}",
                      confidence=float(min(1.0, r.support / 1000.0)))
            for _, r in shared.iterrows()])
        return shared
