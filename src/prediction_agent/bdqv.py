"""
BDQVs - BOM Data Quality Violations.

A BDQV is a *typed, localised, timestamped* assertion about a part at a BOM
maturity phase. It replaces the single `erroneous` scalar of the original
pipeline with a vocabulary, so that patterns can be named, counted, reused
across vehicle generations and explained to an engineer.

Every detector here works only on columns that exist in the published,
anonymised dataset (component, part, timestamp, feature_1..6, rho_v, anom) and
never on the label. Detectors are deliberately simple and rule-based: they are
the agent's perception layer, and they must be auditable by a BOM engineer.
"""

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

# Closed taxonomy. Adding a type here is the only way to extend the vocabulary,
# which keeps the interface between agents stable.
# Types that are high-prevalence *flags* rather than violations. They stay in
# the model's feature matrix but are excluded from the pattern signature:
# feature_1 (the delayed-part indicator) is set on 81 % of all rows, so
# including it in signatures produced one dominant, uninformative pattern
# ("delayed_part|variant_churn", support 235k, lift 1.03) that drowned out the
# selective combinations the registry exists to find.
FLAG_TYPES = ("delayed_part",)

BDQV_TYPES = (
    "rho_v_deviation",      # structural signature far from its component cohort
    "rho_v_phase_jump",     # structural signature moved sharply between phases
    "structural_anomaly",   # isolation forest flag
    "delayed_part",         # part integrated later than initially planned
    "quantity_outlier",     # implausible quantity relative to the cohort
    "variant_churn",        # part reassigned across many components in a phase
    "newly_introduced",     # part absent in the previous phase
    "single_phase_part",    # part occurs in exactly one phase (orphan)
)


# The selective types a pattern signature is built from.
SIGNATURE_TYPES = tuple(t for t in BDQV_TYPES if t not in FLAG_TYPES)


@dataclass
class BDQV:
    """One typed violation. `evidence` is what the agent shows the engineer."""
    type: str
    component: float
    part: float
    timestamp: float
    severity: float
    evidence: dict[str, float] = field(default_factory=dict)
    source: str = "detector"

    @property
    def id(self) -> str:
        return (f"bdqv:{self.type}:c{self.component:g}"
                f":p{self.part:g}:t{self.timestamp:g}")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["id"] = self.id
        return out


def _z(series: pd.Series) -> pd.Series:
    std = series.std()
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def _squash(x: pd.Series, scale: float) -> pd.Series:
    """Map an unbounded deviation onto a (0, 1) severity."""
    return (2.0 / (1.0 + np.exp(-np.abs(x) / scale))) - 1.0


def derive_bdqv_frame(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Vectorised BDQV derivation.

    Returns one row per input row with a severity column per BDQV type
    (0.0 where the violation does not fire). A long-form list of `BDQV`
    objects for individual parts is produced on demand by `bdqvs_for_rows`,
    because materialising 350k dataclasses is pointless for training.
    """
    out = pd.DataFrame(index=df.index)

    # --- rho_v deviation within the (component, phase) cohort -------------
    cohort = df.groupby(["component", "timestamp"])["rho_v"]
    z_cohort = cohort.transform(_z).fillna(0.0)
    out["rho_v_deviation"] = np.where(
        z_cohort.abs() >= z_threshold, _squash(z_cohort, 3.0), 0.0)

    # --- rho_v jump between consecutive phases of the same part ----------
    ordered = df.sort_values(["part", "timestamp"])
    prev = ordered.groupby("part")["rho_v"].shift(1)
    jump = (ordered["rho_v"] - prev).reindex(df.index)
    jump_z = _z(jump.fillna(0.0))
    out["rho_v_phase_jump"] = np.where(
        jump_z.abs() >= z_threshold, _squash(jump_z, 3.0), 0.0)

    # --- isolation forest flag -------------------------------------------
    if "anom" in df.columns:
        out["structural_anomaly"] = np.where(df["anom"] == -1, 0.6, 0.0)
    else:
        out["structural_anomaly"] = 0.0

    # --- late integration -------------------------------------------------
    # The paper's figures use feature_1 as the delayed-delivery indicator.
    # feature_1 is the binary delayed-delivery indicator (the paper's `rldd`).
    # Error rate is 0.158 when set vs. 0.136 when not - a real but weak signal,
    # so it gets a low severity and is treated as a flag (see FLAG_TYPES).
    if "feature_1" in df.columns:
        out["delayed_part"] = np.where(df["feature_1"] > 0, 0.2, 0.0)
    else:
        out["delayed_part"] = 0.0

    # --- quantity outlier -------------------------------------------------
    if "feature_5" in df.columns:
        z5 = df.groupby("component")["feature_5"].transform(_z).fillna(0.0)
        out["quantity_outlier"] = np.where(
            z5.abs() >= z_threshold, _squash(z5, 3.0), 0.0)
    else:
        out["quantity_outlier"] = 0.0

    # --- variant churn: one part across many components in one phase -----
    churn = (df.groupby(["timestamp", "part"])["component"]
               .transform("nunique").astype(float))
    # `churn > 1` fired on 100 % of rows: in a multi-level BOM a part is
    # normally shared across components, so sharing is the baseline, not the
    # violation. Fire only on the upper tail of the churn distribution.
    churn_cut = max(3.0, float(churn.quantile(0.95)))
    out["variant_churn"] = np.where(
        churn >= churn_cut, np.minimum(churn / (2 * churn_cut), 1.0), 0.0)

    # --- newly introduced / single-phase parts ---------------------------
    # "Newly introduced" must mean "absent from the previous phase", not
    # "first phase in this slice" - the latter fires on everything in phase 0
    # of a training window and on nothing in a later test window, i.e. it is
    # not comparable between splits.
    phases_sorted = np.sort(df["timestamp"].unique())
    prev_of = {p: (phases_sorted[i - 1] if i else None)
               for i, p in enumerate(phases_sorted)}
    present = set(zip(df["part"].to_numpy(), df["timestamp"].to_numpy()))
    prev_phase = df["timestamp"].map(prev_of)
    was_present = [
        (prev is not None) and ((part, prev) in present)
        for part, prev in zip(df["part"].to_numpy(), prev_phase.to_numpy())]
    out["newly_introduced"] = np.where(
        (~np.asarray(was_present)) & prev_phase.notna().to_numpy(), 0.4, 0.0)
    n_phases = df.groupby("part")["timestamp"].transform("nunique")
    out["single_phase_part"] = np.where(n_phases == 1, 0.5, 0.0)

    out = out[list(BDQV_TYPES)].astype(float)
    out["bdqv_count"] = (out > 0).sum(axis=1)
    out["bdqv_severity_max"] = out[list(BDQV_TYPES)].max(axis=1)
    out["bdqv_severity_sum"] = out[list(BDQV_TYPES)].sum(axis=1)
    return out


def bdqv_signature(bdqv_frame: pd.DataFrame) -> pd.Series:
    """
    A canonical, hashable signature of which BDQV types fired for a row.

    This is the key the pattern registry is built on: a signature such as
    "late_integration|rho_v_deviation" is the reusable, named pattern.
    """
    active = bdqv_frame[list(SIGNATURE_TYPES)] > 0
    types = np.array(SIGNATURE_TYPES)
    return pd.Series(
        ["|".join(types[row]) if row.any() else "clean"
         for row in active.to_numpy()],
        index=bdqv_frame.index, name="bdqv_signature")


def bdqvs_for_rows(df: pd.DataFrame, bdqv_frame: pd.DataFrame,
                   index) -> list[BDQV]:
    """Materialise BDQV objects for selected rows (explanations, KB writes)."""
    result: list[BDQV] = []
    for idx in index:
        row, sev = df.loc[idx], bdqv_frame.loc[idx]
        for bdqv_type in BDQV_TYPES:
            if sev[bdqv_type] <= 0:
                continue
            result.append(BDQV(
                type=bdqv_type,
                component=float(row["component"]),
                part=float(row["part"]),
                timestamp=float(row["timestamp"]),
                severity=float(sev[bdqv_type]),
                evidence={"rho_v": float(row.get("rho_v", np.nan)),
                          "anom": float(row.get("anom", 0))},
            ))
    return result
