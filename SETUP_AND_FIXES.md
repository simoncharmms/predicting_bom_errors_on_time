# Setup, reproduction and applied fixes

The repository as published does not run on any current Python/pandas/scikit-learn
stack. This file documents exactly what was needed to get the pipeline green
end to end, and which defects were found on the way.

## Reproduce

```bash
git clone https://github.com/simoncharmms/predicting_bom_errors_on_time.git
cd predicting_bom_errors_on_time
uv venv --python 3.11 .venv          # TensorFlow 2.15 requires Python <=3.11 and numpy<2
uv pip install --python .venv/bin/python -r requirements.txt
./run.sh                             # wrapper: sets PYTHONPATH=src, MPLBACKEND=Agg
```

Useful overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLP_EPOCHS` | 510 | short smoke runs (`MLP_EPOCHS=5 ./run.sh`) |
| `MLP_BATCH_SIZE` | 64 | |
| `HC_MAX_SAMPLE` | 5000 | cap on the subsample fed to hierarchical linkage |

`SAMPLE = True` in `src/constants.py` keeps the Procrustes stage on a 1 % sample.

## Blockers fixed to make it run

1. **`run.sh` / import path.** `main.py` uses flat imports (`from constants import ...`)
   while the modules live in `src/`. Without `PYTHONPATH=src` nothing imports.
2. **`model.partile(...)` → `model.compile(...)`** in `multi_output_mlp.py`. A
   botched global find/replace of `comp`→`part` broke the only call that matters
   (the same replace also corrupted comments: "partute", "partare", "partile").
3. **`Series.append` removed in pandas 2.0** (`utils.create_adjacency`) →
   `pd.concat`. This was the fatal one: it raised inside the loop's bare
   `except: pass`, so the Procrustes stage silently produced **zero** results
   and only failed three steps later with a confusing `KeyError`.
4. **`rho_v` merge** rewritten (see defect #1 below).
5. **Hierarchical linkage OOM.** `linkage()` on a full component subset asked for
   1007 GiB. Now capped via `HC_MAX_SAMPLE` (the linkage result is only used to
   pick a cluster count).
6. **matplotlib 3.x 3-D API.** `Axes3D(fig, rect=...)`, `ax.w_xaxis`, `ax.dist`
   were removed → `fig.add_subplot(projection='3d')`, `set_xticklabels`.
7. **`plt.close` without parentheses** — never closed anything, so the clustering
   loop leaked figures until matplotlib warned. → `plt.close(fig)`.
8. **Isolation forest referenced 8 columns that do not exist** in the published
   anonymized data (`lcim`, `indc`, `istp_bool`, `nlbw`, `scop`, `rldd_bool`,
   `sca`, `conf`) and `max_features=14` exceeded the real feature count.
9. **sklearn metric dtype errors.** `erroneous` is `float64`; `accuracy_score` /
   `confusion_matrix` reject float 0.0/1.0 as "continuous". Added explicit casts.
10. Added `random_state` / `n_init` to `KMeans` and `IsolationForest` so runs are
    reproducible at all.

## Substantive defects found (not just compatibility)

### 1. The `rho_v` merge inflated the dataset 14x
Original:
```python
df_rho_v["mapping"] = df_rho_v.index.astype(str)          # index = timestamp + part, as strings
df["mapping"] = df["timestamp"].astype(str) + df["part"].astype(str)
df = pd.merge(df, df_rho_v, on="mapping", how="left")
```
The key is a **string concatenation**, so `("1.0", "12.0")` and `("1.01", "2.0")`
collide, and the right-hand table has duplicate keys (a part recurs across
components). Result: 350,263 rows became **4,917,477 rows** of duplicated
records, which then flowed into clustering, anomaly detection and the MLP —
i.e. the same physical part appears many times across the train/test split.
Replaced with a tidy `(timestamp, part) -> mean(rho_v)` table merged on two
typed keys, plus a row-count assertion.

### 2. Label leakage into the "unsupervised" anomaly detector
`to_model_columns = metrics_df.columns` handed `erroneous` (the target) and a
reset `index` column straight to `IsolationForest`. Now excluded.

### 3. `except: pass` around the entire Procrustes inner loop
This hid defect #3 above completely. Any future change to the adjacency code
will fail silently in the same way. It should catch `LinAlgError` / `ValueError`
only, and count the skips.

### 4. `rho_v = np.sqrt(np.trace(R.T * R))`
`R` is created via `np.mat(...)` upstream, so `*` is matrix multiplication and
this happens to be right — but only by accident. On a plain `ndarray` it is
elementwise and the metric is silently wrong. `np.mat` is deprecated; this is a
latent bug. Use `np.sqrt(np.trace(R.T @ R))` on ndarrays.

### 5. Train/test split ignores time
`train_test_split(..., random_state=1)` shuffles rows, but the model predicts
*when* an error appears. A random split lets the model see later maturity phases
of the same part while predicting earlier ones. The evaluation should be split
by `timestamp` (or grouped by `part`).

### 6. Accuracy on a 0.24 % positive class
With `MLP_EPOCHS=5` the run reports ~0.55 accuracy, recall ~0.73, precision
~0.004 — 157k false positives. The earlier 510-epoch-style configuration
collapses to predicting the majority class and reports 0.998 accuracy with
**zero** true positives. Accuracy is the wrong headline metric here; PR-AUC and
precision@k are what a BOM engineer actually experiences.

### 7. Chi-squared / t-test stage returns all-NaN
Every `ttest_ind` in `chi_squared_test.py` returns NaN ("sample too small") on
the sampled data, so the "feature engineering" step currently informs nothing.

---

# Round 2: keeping the approach, fixing the protocol

The pipeline architecture is unchanged (Procrustes → chi² → k-means → isolation
forest → multi-output MLP). Only the label definition, the leakage paths, the
training protocol and the metrics were corrected. Two of these change the
problem itself.

## 8. `erroneous` is not a binary label (most consequential finding)

```
dtype float64   min 0.0   max 1.0456   distinct values 157
296,380 rows are exactly 0.0   |   53,883 rows are non-zero
```

The column is a continuous anonymisation artifact, not an indicator. The
original code binarised it with `df[target_clas].astype(int)`, and `astype(int)`
**truncates**: only the 868 rows whose value happens to exceed 1.0 became
positives. The frequently quoted "0.24 % error rate" is therefore a casting
artifact — it discards 98.4 % of the labelled errors.

Binarising at `> 0` (now `LABEL_THRESHOLD` in `constants.py`) gives **53,883
positives, 15.4 %**. That turns an unlearnable extreme-imbalance problem into an
ordinary moderately imbalanced one. Anyone reproducing the paper's numbers
should check this first.

## 9. The k-means "contextualisation" leaked the label

```python
df_hc = df_sub[["part", "erroneous", "rho_v"]]   # clusters on the target
...
est = KMeans(...); est.fit(X); df["clus"] = labels
...
features = [..., 'clus', 'anom']                 # handed to the MLP
```

The cluster id was computed *from* `erroneous` and then used as an input
feature. Combined with the isolation-forest leak (§2), the model was reading a
compressed copy of its own target. With the leak in place validation PR-AUC was
0.53; after removing it, 0.17–0.20. The reported "optimisation through
contextualisation via k-means" is largely this artifact. `CLUSTER_FEATURES` is
now `["part", "rho_v", "timestamp"]`.

## 10. Training protocol

| Defect | Fix |
| --- | --- |
| No feature scaling; `feature_5 ≈ -91`, `rho_v ≈ 1` fed to a ReLU MLP | `StandardScaler`, fit on train rows only |
| `Dense(2, activation='sigmoid')` under `sparse_categorical_crossentropy` — outputs are not a distribution | `activation='softmax'` |
| Random `train_test_split` on a task defined as predicting *when* | split on cumulative row share of maturity phases (train ≤60 %, val 60–80 %, test >80 %) |
| 510 epochs, no validation set, no early stopping | validation split + `EarlyStopping(restore_best_weights=True)` |
| Imbalance unaddressed | per-sample weights on the classification head (Keras rejects `class_weight` for multi-output) |
| Threshold hard-coded to 0.5 via `np.around` | threshold chosen on validation under an explicit alert budget (`ALERT_BUDGET_FRACTION`) |
| Metrics reported on data the model trained on | train / val / test reported separately; a `split` column is written to the prediction CSV |
| Headline metric = accuracy on an imbalanced target | PR-AUC, ROC-AUC, precision@k, recall@k, lift over base rate, plus phase MAE; written to `data/mlp_report.json` |
| `contamination=0.05` vs. a real anomaly rate 20x lower | `contamination="auto"`, overridable via `IFOREST_CONTAMINATION` |
| Nothing seeded | `RANDOM_SEED` through `KMeans`, `IsolationForest`, numpy, `random`, TensorFlow |
| Naive phase split left 199 rows / 0 positives in test | row-share-based phase split (train 216,989 / val 110,043 / test 23,231) |

## Honest baseline after all fixes

Both leaks removed, label corrected, temporal split, 60 epochs with early
stopping:

| | validation | test |
| --- | --- | --- |
| rows | 110,043 | 23,231 |
| base rate | 0.155 | 0.173 |
| PR-AUC | 0.17 – 0.20 | 0.24 – 0.26 |
| ROC-AUC | 0.55 – 0.57 | 0.62 – 0.63 |
| precision@1 % | 0.10 – 0.21 | 0.25 – 0.28 |
| phase MAE | 1.9 – 2.7 phases | 5.4 – 5.8 phases |

Read this plainly: the approach retains **modest but real** signal — roughly
1.4–1.6x lift over the base rate at a 1 % alert budget on the held-out later
phases — and it is unstable between runs. The phase-regression head is not
usable (an error of 5 phases on a 14-phase scale). The previously impressive
figures came from the label-truncation artifact plus two leakage paths, not from
the model. This is the baseline the Prediction Agent has to beat, and the
ablation ladder in `PREDICTION_AGENT_DESIGN.md` §5 is how to prove it does.
