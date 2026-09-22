"""
Configuration-behaviour layer.

IMPORTANT, PLEASE READ BEFORE INTERPRETING ANY RESULT
-----------------------------------------------------
The published dataset contains no configuration event log - it is a BOM
snapshot. This module therefore does two separate things:

  1. `ConfigEventLog` defines the real interface the agent expects from a
     productive PLM/configurator system (one row per user interaction).
  2. `simulate_event_log` generates a *stand-in* log so the agent is runnable
     end to end on the published data.

The simulator is deliberately derived from BOM structure ONLY and never from
the `erroneous` label. That means it cannot inject artificial predictive signal,
and the honest expected outcome is that the behaviour branch contributes close
to nothing on this dataset. Any real uplift has to come from a real event log.
Reporting a gain from a label-derived simulator would be self-deception, which
is why the evaluation in `evaluate.py` runs the behaviour branch both as-is and
with its features permuted.

Privacy by construction: features are z-scored against each user's own
baseline, so the model learns a *behaviour type* rather than ranking people.
Only cohort aggregates are persisted. This is a works-council requirement in a
German OEM setting and also the better modelling choice, because relative
deviation predicts error better than absolute speed.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# The schema a productive integration must deliver.
CONFIG_EVENT_COLUMNS = (
    "session_id", "user_id", "role", "timestamp", "component", "part",
    "action", "dwell_ms", "undo_count", "help_opened", "variant_switches",
    "copy_from_generation",
)

BEHAVIOUR_FEATURES = (
    "beh_dwell_z",              # dwell time vs. the user's own median
    "beh_undo_rate",
    "beh_variant_switches",
    "beh_help_opened",
    "beh_copy_depth",
    "beh_session_load",         # parts touched in the session
    "beh_role_novice",
    "beh_edits_per_part",
)


def simulate_event_log(df: pd.DataFrame, n_users: int = 40,
                       seed: int = 1) -> pd.DataFrame:
    """
    Structure-only stand-in event log. Uses component/part/phase/rho_v and a
    deterministic RNG; never touches `erroneous`.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    # Sessions are (component, phase) working sets - how engineers actually work.
    session_key = (df["component"].astype(str) + "_"
                   + df["timestamp"].astype(str))
    session_codes = pd.factorize(session_key)[0]
    user_id = session_codes % n_users
    tenure = rng.random(n_users)                      # per-user competence prior
    base_dwell = 400 + 2600 * tenure[user_id]

    complexity = (df.groupby(["component", "timestamp"])["part"]
                    .transform("nunique").to_numpy(dtype=float))
    complexity = complexity / max(complexity.max(), 1.0)

    log = pd.DataFrame({
        "session_id": session_codes,
        "user_id": user_id,
        "role": np.where(tenure[user_id] < 0.25, "novice", "engineer"),
        "timestamp": df["timestamp"].to_numpy(),
        "component": df["component"].to_numpy(),
        "part": df["part"].to_numpy(),
        "action": rng.choice(["edit", "assign", "replace"], size=n,
                             p=[0.6, 0.3, 0.1]),
        "dwell_ms": np.clip(rng.lognormal(np.log(base_dwell), 0.6)
                            * (0.5 + complexity), 50, None),
        "undo_count": rng.poisson(0.2 + 1.2 * complexity),
        "help_opened": (rng.random(n) < 0.05 + 0.1 * complexity).astype(int),
        "variant_switches": rng.poisson(0.3 + 1.0 * complexity),
        "copy_from_generation": rng.integers(0, 3, size=n),
    })
    return log[list(CONFIG_EVENT_COLUMNS)]


def behaviour_features(log: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """
    Aggregate an event log into per-row behaviour features.

    Dwell time is z-scored **within each user**, so the feature expresses "this
    engineer hesitated more than they usually do", not "this engineer is slow".
    """
    out = pd.DataFrame(index=index)
    grp = log.groupby("user_id")["dwell_ms"]
    med = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median()).replace(0, np.nan)
    out["beh_dwell_z"] = ((log["dwell_ms"] - med) / mad).fillna(0.0).to_numpy()

    edits = log.groupby("session_id")["part"].transform("size").astype(float)
    out["beh_undo_rate"] = (log["undo_count"] / edits.clip(lower=1)).to_numpy()
    out["beh_variant_switches"] = log["variant_switches"].to_numpy(dtype=float)
    out["beh_help_opened"] = log["help_opened"].to_numpy(dtype=float)
    out["beh_copy_depth"] = log["copy_from_generation"].to_numpy(dtype=float)
    out["beh_session_load"] = (log.groupby("session_id")["part"]
                                 .transform("nunique").to_numpy(dtype=float))
    out["beh_role_novice"] = (log["role"] == "novice").to_numpy(dtype=float)
    out["beh_edits_per_part"] = edits.to_numpy()
    return out[list(BEHAVIOUR_FEATURES)].astype(float)


class OnlineBehaviourModel:
    """
    Small incrementally-updated model so the agent adapts *within* a session,
    while the heavy structural model stays batch. Logistic SGD with
    `partial_fit`; the agent calls `update` from engineer feedback.
    """

    def __init__(self, seed: int = 1):
        self.scaler = StandardScaler()
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-4,
                                 random_state=seed)
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OnlineBehaviourModel":
        Xs = self.scaler.fit_transform(X)
        self.clf.partial_fit(Xs, y, classes=np.array([0, 1]))
        self.fitted = True
        return self

    def update(self, X: pd.DataFrame, y: np.ndarray) -> None:
        if not self.fitted:
            self.fit(X, y)
            return
        self.clf.partial_fit(self.scaler.transform(X), y)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(X))
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]
