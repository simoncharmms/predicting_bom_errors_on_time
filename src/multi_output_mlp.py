"""
Multi-output MLP predicting (a) whether a part is erroneous and (b) at which
BOM maturity phase the error occurs.

This is the same model family as in the paper (one shared trunk, one regression
head for the timestamp, one classification head for `erroneous`), but the
training and evaluation protocol has been corrected. See SETUP_AND_FIXES.md for
the full list; the changes that affect reported numbers are:

  * features are standardised (fit on train only) - the raw features span
    orders of magnitude (feature_5 has values around -91, rho_v around 1),
    which an unscaled ReLU MLP cannot handle;
  * the classification head uses softmax, not sigmoid - sigmoid outputs do not
    form a distribution and are invalid under sparse_categorical_crossentropy;
  * the split is temporal (early maturity phases -> late), not random, because
    the task is to predict errors *on time*;
  * the positive class (0.24 %) is weighted, so the model cannot win by
    predicting "no error" everywhere;
  * the decision threshold is chosen on a validation split under an explicit
    alert budget instead of being hard-coded to 0.5 via np.around();
  * the headline metrics are PR-AUC, precision@k and recall@k, not accuracy.
"""

### ---------------------------------------------------------------------------
### Preliminaries.
### ---------------------------------------------------------------------------

import os
import time
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (average_precision_score, confusion_matrix,
                            mean_absolute_error, precision_score,
                            recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from utils import replacegaps
from constants import (FP_DATA, RANDOM_SEED, ALERT_BUDGET_FRACTION,
                       LABEL_THRESHOLD)

start_time = time.time()

FEATURES = ['component', 'part', 'feature_1', 'feature_2', 'feature_3',
            'feature_4', 'feature_5', 'rho_v', 'clus', 'anom']
TARGET_REG = 'timestamp'
TARGET_CLAS = 'erroneous'


def set_seeds(seed=RANDOM_SEED):
    """Make a run reproducible. The original code seeded nothing."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def temporal_split(df, val_fraction=0.2, test_fraction=0.2):
    """
    Split by BOM maturity phase instead of shuffling rows.

    A random split lets the model observe later maturity phases of the very same
    part while it is asked to predict an earlier one, which is exactly the
    information a productive system does not have. Phases are ordered, so the
    latest phases become the test set.
    """
    # Split on cumulative ROW share, not on the count of distinct phases: the
    # phases are wildly unbalanced (phases 0-4 hold 93 % of all rows), so
    # "last 20 % of phases" left only 199 rows and zero positives in test.
    counts = df[TARGET_REG].value_counts().sort_index()
    phases = counts.index.to_numpy()
    share = counts.cumsum() / counts.sum()
    train_end = 1.0 - val_fraction - test_fraction
    val_end = 1.0 - test_fraction
    prev = share.shift(1).fillna(0.0).to_numpy()
    val_phases = phases[(prev >= train_end) & (prev < val_end)]
    test_phases = phases[prev >= val_end]
    is_test = df[TARGET_REG].isin(test_phases)
    is_val = df[TARGET_REG].isin(val_phases)
    return ~(is_test | is_val), is_val, is_test, val_phases, test_phases


def precision_recall_at_k(y_true, scores, k):
    """
    Precision and recall inside an alert budget of k parts.

    This is what a BOM engineer actually experiences: they work a ranked
    worklist of fixed length, they never see a 0.5 threshold.
    """
    k = int(min(k, len(scores)))
    if k == 0:
        return float('nan'), float('nan'), 0
    order = np.argsort(-np.asarray(scores))[:k]
    hits = int(np.asarray(y_true)[order].sum())
    total_pos = int(np.asarray(y_true).sum())
    precision = hits / k
    recall = hits / total_pos if total_pos else float('nan')
    return precision, recall, hits


def threshold_for_budget(scores, budget_fraction):
    """Pick the score threshold that emits `budget_fraction` of rows as alerts."""
    return float(np.quantile(scores, 1.0 - budget_fraction))


def multi_output_mlp():
    set_seeds()

    ### -----------------------------------------------------------------------
    ### Load data.
    ### -----------------------------------------------------------------------
    df = pd.read_csv(FP_DATA + 'bom_data_clus_anom.csv', sep=',',
                     encoding='latin-1')
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = replacegaps(df).fillna(0)

    ### -----------------------------------------------------------------------
    ### Hyperparameters.
    ### -----------------------------------------------------------------------
    epochs = int(os.environ.get("MLP_EPOCHS", 200))
    batch_size = int(os.environ.get("MLP_BATCH_SIZE", 256))
    patience = int(os.environ.get("MLP_PATIENCE", 10))

    ### -----------------------------------------------------------------------
    ### Temporal split.
    ### -----------------------------------------------------------------------
    tr, va, te, val_phases, test_phases = temporal_split(df)
    print(f"Split by maturity phase -> train {tr.sum()} | val {va.sum()} "
          f"| test {te.sum()}")
    print(f"  validation phases: {val_phases.tolist()}")
    print(f"  test phases:       {test_phases.tolist()}")

    X_all = df[FEATURES].astype(float)
    y_reg_all = df[TARGET_REG].astype(int)
    # Binarise at > LABEL_THRESHOLD. `astype(int)` would truncate the
    # continuous anonymised label and discard 98 % of the true positives.
    y_clas_all = (df[TARGET_CLAS] > LABEL_THRESHOLD).astype(int)

    # Standardise. Fit on the training rows only - fitting on everything leaks
    # test-set scale information into training.
    scaler = StandardScaler().fit(X_all[tr])
    Xtr, Xva, Xte = (scaler.transform(X_all[m]) for m in (tr, va, te))

    ytr_reg, yva_reg, yte_reg = (y_reg_all[m].values for m in (tr, va, te))
    ytr_clas, yva_clas, yte_clas = (y_clas_all[m].values for m in (tr, va, te))

    base_rate = ytr_clas.mean()
    print(f"Positive class rate in train: {base_rate:.5f} "
          f"({ytr_clas.sum()} of {len(ytr_clas)})")

    ### -----------------------------------------------------------------------
    ### Model.
    ### -----------------------------------------------------------------------
    n_features = len(FEATURES)
    visible = Input(shape=(n_features,), name='features')
    hidden1 = Dense(64, activation='relu', kernel_initializer='he_normal')(visible)
    hidden2 = Dense(32, activation='relu', kernel_initializer='he_normal')(hidden1)
    out_reg = Dense(1, activation='linear', name='phase')(hidden2)
    # softmax, not sigmoid: sparse_categorical_crossentropy expects a
    # distribution over the two classes.
    out_clas = Dense(2, activation='softmax', name='erroneous')(hidden2)

    model = Model(inputs=visible, outputs=[out_reg, out_clas])
    model.compile(
        loss={'phase': 'mse', 'erroneous': 'sparse_categorical_crossentropy'},
        loss_weights={'phase': 1.0, 'erroneous': 1.0},
        optimizer='adam',
    )

    # Keras does not accept class_weight for multi-output models, so the
    # imbalance is handled with per-sample weights on the classification head.
    pos_weight = (1.0 - base_rate) / max(base_rate, 1e-9)
    w_clas_tr = np.where(ytr_clas == 1, pos_weight, 1.0)
    w_clas_va = np.where(yva_clas == 1, pos_weight, 1.0)
    print(f"Positive-class weight: {pos_weight:.1f}")

    early = EarlyStopping(monitor='val_loss', patience=patience,
                          restore_best_weights=True)
    model.fit(
        Xtr, {'phase': ytr_reg, 'erroneous': ytr_clas},
        sample_weight={'phase': np.ones_like(ytr_reg, dtype=float),
                       'erroneous': w_clas_tr},
        validation_data=(Xva, {'phase': yva_reg, 'erroneous': yva_clas},
                         {'phase': np.ones_like(yva_reg, dtype=float),
                          'erroneous': w_clas_va}),
        epochs=epochs, batch_size=batch_size, callbacks=[early], verbose=2,
    )

    ### -----------------------------------------------------------------------
    ### Evaluation.
    ### -----------------------------------------------------------------------
    def score(X):
        reg, clas = model.predict(X, batch_size=4096, verbose=0)
        return reg.ravel(), clas[:, 1]

    reg_va, p_va = score(Xva)
    reg_te, p_te = score(Xte)

    # The operating point is chosen on validation, never on test.
    threshold = threshold_for_budget(p_va, ALERT_BUDGET_FRACTION)
    print(f"\nAlert budget {ALERT_BUDGET_FRACTION:.1%} -> "
          f"threshold {threshold:.4f} (chosen on validation)")

    report = {}
    for name, yc, yr, p, r in (("validation", yva_clas, yva_reg, p_va, reg_va),
                               ("test", yte_clas, yte_reg, p_te, reg_te)):
        k = int(round(len(p) * ALERT_BUDGET_FRACTION))
        prec_k, rec_k, hits = precision_recall_at_k(yc, p, k)
        yhat = (p >= threshold).astype(int)
        block = {
            "n": int(len(p)),
            "positives": int(yc.sum()),
            "base_rate": float(yc.mean()),
            # PR-AUC is the threshold-free headline metric for a 0.24 %
            # positive rate; accuracy is reported only for comparability.
            "pr_auc": float(average_precision_score(yc, p)) if yc.sum() else None,
            "roc_auc": float(roc_auc_score(yc, p)) if yc.sum() else None,
            "lift_over_base": (float(prec_k / yc.mean())
                               if yc.sum() and yc.mean() > 0 else None),
            "alert_budget_k": k,
            "precision_at_k": float(prec_k),
            "recall_at_k": float(rec_k),
            "hits_at_k": hits,
            "precision_at_threshold": float(precision_score(yc, yhat,
                                                            zero_division=0)),
            "recall_at_threshold": float(recall_score(yc, yhat,
                                                      zero_division=0)),
            "accuracy": float((yhat == yc).mean()),
            "phase_mae": float(mean_absolute_error(yr, r)),
            "confusion_matrix": confusion_matrix(yc, yhat).tolist(),
        }
        report[name] = block
        print(f"\n--- {name} ---")
        for key, value in block.items():
            print(f"  {key}: {value}")

    ### -----------------------------------------------------------------------
    ### Persist predictions and the report.
    ### -----------------------------------------------------------------------
    reg_all, p_all = score(scaler.transform(X_all))
    out = df[FEATURES].copy()
    out["mlpp_reg"] = reg_all
    out["mlpp_score"] = p_all
    out["mlpp_clas"] = (p_all >= threshold).astype(int)
    out["erroneous"] = y_clas_all
    out["split"] = np.select([tr, va, te], ["train", "val", "test"],
                             default="unassigned")
    out["conf"] = np.select(
        [(out.erroneous == 1) & (out.mlpp_clas == 1),
         (out.erroneous == 0) & (out.mlpp_clas == 1),
         (out.erroneous == 1) & (out.mlpp_clas == 0)],
        ["true positive", "false positive", "false negative"],
        default="true negative")
    out.to_csv(FP_DATA + f"mlp_{epochs}_{batch_size}pred.csv", index=False)

    report["label_threshold"] = LABEL_THRESHOLD
    report["threshold"] = threshold
    report["alert_budget_fraction"] = ALERT_BUDGET_FRACTION
    report["positive_class_weight"] = pos_weight
    with open(FP_DATA + "mlp_report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {FP_DATA}mlp_report.json")
    return report


### ---------------------------------------------------------------------------
### End.
### ---------------------------------------------------------------------------

if __name__ == '__main__':
    multi_output_mlp()
    print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
