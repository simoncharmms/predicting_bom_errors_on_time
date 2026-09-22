FP_DATA = 'data/'
SAMPLE = True

# Reproducibility: the original code seeded nothing, so no run was repeatable.
RANDOM_SEED = 1

# Share of rows that may be emitted as alerts. A prescriptive system is only
# adopted if the worklist is short, so the operating point is a budget rather
# than an implicit 0.5 threshold.
ALERT_BUDGET_FRACTION = 0.01

# Cap on the subsample fed to hierarchical linkage (O(n^2) memory).
HC_MAX_SAMPLE = 5000

# The published `erroneous` column is NOT a 0/1 indicator: it is a continuous
# anonymisation artifact in [0, 1.046] with 157 distinct values. 296,380 of
# 350,263 rows are exactly 0.0. The original code binarised it with
# `astype(int)`, which TRUNCATES, so only the 868 rows with a value >= 1.0 were
# treated as errors - an apparent positive rate of 0.24 % that is a casting
# artifact rather than a property of the data. Binarising at "> 0" instead
# yields 53,883 positives (15.4 %), which is the intended label.
LABEL_THRESHOLD = 0.0

# Features used for k-means "contextualisation". The original code clustered on
# ["part", "erroneous", "rho_v"] - i.e. on the target - and passed the cluster
# id to the MLP as a feature, leaking the label. Observable attributes only.
CLUSTER_FEATURES = ["part", "rho_v", "timestamp"]
