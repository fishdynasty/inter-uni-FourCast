"""
Credit Card Default Prediction — Full Reproducible Pipeline
Inter-Uni Datathon (FourCast)

This script reproduces the exact leaderboard submission:
    submission_tuned_ensemble.csv

It consolidates the work done across three notebooks into one linear,
runnable script:
    1. Data_clean.ipynb  -> data quality checks (no transformations were
                             required: no missing values, no duplicates)
    2. EDA.ipynb         -> exploratory analysis (not required to
                             regenerate the submission; kept as a separate
                             notebook, not re-run here)
    3. Modelling.ipynb   -> feature engineering, model selection, tuning,
                             ensembling, and final prediction (this is
                             what is reproduced below)

Pipeline stages, in order:
    1. Load competition data
    2. Data-quality checks (mirrors Data_clean.ipynb; no cleaning
       transformations were applied because none were needed)
    3. Feature engineering
    4. Model validation (5-fold CV) — for reference only, not required
       to produce the final file, but included so the reported CV scores
       in the methodology report can be regenerated
    5. Final model training on 100% of training data
    6. Test-set inference
    7. Weighted blend (post-processing)
    8. Write final submission file

Run with:
    python full_pipeline.py

Expected runtime: a few minutes on a laptop CPU (CatBoost is the slower
of the two models).
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss
from sklearn.base import clone

from scipy.optimize import minimize_scalar

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

RANDOM_SEED = 50

# ---------------------------------------------------------------------------
# 1. Load competition data
# ---------------------------------------------------------------------------
# Expected folder layout (relative to this script):
#   ../data/raw/train.csv
#   ../data/raw/test.csv
# Adjust the paths below if your folder layout differs.

train = pd.read_csv("../data/raw/train.csv")
test = pd.read_csv("../data/raw/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ---------------------------------------------------------------------------
# 2. Data-quality checks (mirrors Data_clean.ipynb)
# ---------------------------------------------------------------------------
# No missing values, no duplicate rows, and no duplicate client_ids were
# found in either train or test, so no cleaning/imputation step is applied.

assert train.isna().sum().sum() == 0, "Unexpected missing values in train"
assert test.isna().sum().sum() == 0, "Unexpected missing values in test"
assert train.duplicated().sum() == 0, "Unexpected duplicate rows in train"
assert train["client_id"].duplicated().sum() == 0, "Duplicate client_id in train"
assert test["client_id"].duplicated().sum() == 0, "Duplicate client_id in test"

X = train.drop(columns=["default", "client_id"])
y = train["default"]
X_test = test.drop(columns=["client_id"])

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
             "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
payment_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

boost_categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]


def add_features(df):
    """Repayment, billing, and utilisation summary features."""
    df = df.copy()

    df["delayed_months"] = (df[pay_cols] > 0).sum(axis=1)
    df["max_delay"] = df[pay_cols].max(axis=1)
    df["avg_repayment_status"] = df[pay_cols].mean(axis=1)

    df["avg_bill"] = df[bill_cols].mean(axis=1)
    df["avg_payment"] = df[payment_cols].mean(axis=1)

    df["recent_utilisation"] = df["BILL_AMT1"] / df["LIMIT_BAL"]

    return df


X_fe = add_features(X)
X_test_fe = add_features(X_test)

# ---------------------------------------------------------------------------
# 4. Model validation (5-fold stratified CV) — for reference / reporting
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# 4a. Logistic Regression baseline (one-hot + standard-scaled features)
categorical_cols = ["SEX", "EDUCATION", "MARRIAGE",
                     "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

from sklearn.preprocessing import StandardScaler  # noqa: E402

logreg_preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

logreg_model = Pipeline([
    ("preprocessor", logreg_preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

logreg_scores = -cross_val_score(logreg_model, X, y, cv=cv, scoring="neg_log_loss")
print("Logistic Regression CV log loss:", logreg_scores.mean())

# 4b. Tuned XGBoost (with engineered features)
xgb_preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), boost_categorical_cols)],
    remainder="passthrough",
)

tuned_xgb_template = Pipeline([
    ("preprocessor", xgb_preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )),
])

xgb_scores = -cross_val_score(tuned_xgb_template, X_fe, y, cv=cv, scoring="neg_log_loss")
print("Tuned XGBoost CV log loss:", xgb_scores.mean())

# 4c. Tuned CatBoost (with engineered features), manual fold loop so native
#     categorical handling (cat_features=) can be used
tuned_xgb_oof = np.zeros(len(X_fe))
tuned_cat_oof = np.zeros(len(X_fe))

for fold, (train_idx, valid_idx) in enumerate(cv.split(X_fe, y), start=1):
    X_train_fold, X_valid_fold = X_fe.iloc[train_idx], X_fe.iloc[valid_idx]
    y_train_fold = y.iloc[train_idx]

    xgb_fold = clone(tuned_xgb_template)
    xgb_fold.fit(X_train_fold, y_train_fold)
    tuned_xgb_oof[valid_idx] = xgb_fold.predict_proba(X_valid_fold)[:, 1]

    cat_fold = CatBoostClassifier(
        iterations=700,
        learning_rate=0.02,
        depth=6,
        loss_function="Logloss",
        verbose=0,
        random_seed=RANDOM_SEED,
        thread_count=-1,
    )
    cat_fold.fit(X_train_fold, y_train_fold, cat_features=boost_categorical_cols)
    tuned_cat_oof[valid_idx] = cat_fold.predict_proba(X_valid_fold)[:, 1]

    print(f"Finished fold {fold}")

print("Tuned XGBoost OOF log loss:", log_loss(y, tuned_xgb_oof))
print("Tuned CatBoost OOF log loss:", log_loss(y, tuned_cat_oof))

# 4d. Find optimal blend weight on out-of-fold predictions
def tuned_blend_loss(w):
    blended = w * tuned_xgb_oof + (1 - w) * tuned_cat_oof
    return log_loss(y, blended)


result = minimize_scalar(tuned_blend_loss, bounds=(0, 1), method="bounded")
tuned_xgb_weight = result.x
tuned_cat_weight = 1 - result.x

print("Optimal XGBoost weight:", tuned_xgb_weight)
print("Optimal CatBoost weight:", tuned_cat_weight)
print("Optimal blend OOF log loss:", result.fun)

# ---------------------------------------------------------------------------
# 5. Final model training on 100% of training data
# ---------------------------------------------------------------------------
final_xgb = Pipeline([
    ("preprocessor", xgb_preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )),
])
final_xgb.fit(X_fe, y)

final_cat = CatBoostClassifier(
    iterations=700,
    learning_rate=0.02,
    depth=6,
    loss_function="Logloss",
    verbose=0,
    random_seed=RANDOM_SEED,
    thread_count=-1,
)
final_cat.fit(X_fe, y, cat_features=boost_categorical_cols)

# ---------------------------------------------------------------------------
# 6. Test-set inference
# ---------------------------------------------------------------------------
xgb_test_pred = final_xgb.predict_proba(X_test_fe)[:, 1]
cat_test_pred = final_cat.predict_proba(X_test_fe)[:, 1]

# ---------------------------------------------------------------------------
# 7. Post-processing: fixed weighted blend
# ---------------------------------------------------------------------------
# Weights below are the optimal weights found via minimize_scalar on the
# out-of-fold predictions in step 4d (fixed here so the exact leaderboard
# file can be regenerated even if optimisation is skipped/re-run differently).
XGB_WEIGHT = 0.2128142851
CAT_WEIGHT = 0.7871857149

final_pred = XGB_WEIGHT * xgb_test_pred + CAT_WEIGHT * cat_test_pred

# ---------------------------------------------------------------------------
# 8. Write final submission file
# ---------------------------------------------------------------------------
submission = pd.DataFrame({
    "client_id": test["client_id"],
    "default": final_pred,
})

submission.to_csv("../submission/submission_tuned_ensemble.csv", index=False)
print(submission.head())
print("Submission written to ../submission/submission_tuned_ensemble.csv")
