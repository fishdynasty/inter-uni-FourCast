# Methodology Report — Credit Card Default Prediction
## Inter-Uni Datathon (Team FourCast)

## 1. Problem and Data

The task is binary classification: predict the probability that a credit
card customer will default, evaluated by binary log loss. The competition
provided:

- `train.csv` — 24,000 labelled customers, 25 columns (24 features + `default`)
- `test.csv` — 6,000 unlabelled customers, 24 columns

Target distribution: 77.88% no default, 22.12% default — moderately
imbalanced. Because the metric is log loss, the priority throughout was
calibrated probabilities rather than raw classification accuracy.

## 2. Data Cleaning and Preprocessing

A full data-quality check was run before any modelling (see
`Data_clean.ipynb`):

- No missing values in train or test
- No duplicate rows in train or test
- No duplicate `client_id` values in train or test
- `SEX`, `EDUCATION`, and `MARRIAGE` are categorical codes rather than
  ordinal numbers; their value counts were inspected for undocumented
  categories (e.g. `EDUCATION` includes categories `0`, `5`, `6` beyond the
  documented 1–4, and `MARRIAGE` includes an undocumented `0`). These
  categories were **kept as-is** rather than dropped or recoded, since
  tree-based models handle them natively and removing rows would have
  discarded real customers.

No imputation, outlier removal, or row filtering was required or applied —
the dataset was clean as provided.

## 3. Exploratory Data Analysis

Key patterns identified (see `EDA.ipynb`):

- **Credit limit (`LIMIT_BAL`)**: right-skewed; default rate decreases
  fairly steadily as credit limit increases (from ~40% in the lowest limit
  band to ~12% in the highest).
- **Repayment status (`PAY_0`–`PAY_6`)**: the strongest single signal in
  the dataset. Default rate rises sharply once a customer has any
  positive repayment-delay code — e.g. for `PAY_0`, default rate is
  ~13% at status ≤0 (no delay) but jumps to 34–75% once a delay is
  present.
- **Demographics (`SEX`, `EDUCATION`, `MARRIAGE`)**: minor differences in
  default rate, much weaker than repayment behaviour.
- **Bill amounts (`BILL_AMT1`–`6`)**: wide range, many extreme values, but
  a weak and non-monotonic relationship with default when grouped.
- **Payment amounts (`PAY_AMT1`–`6`)**: a clearer relationship — customers
  who pay back less recently have a noticeably higher default rate
  (35% at $0 recent payment vs. ~6% above $100k).

These findings directly motivated the feature engineering below.

## 4. Feature Engineering

Starting from the 23 raw predictor columns, six summary features were
added on top of the repayment-status, billing, and payment history
(see `Modelling.ipynb`, `add_features()`):

| Feature | Definition |
|---|---|
| `delayed_months` | Count of the 6 `PAY_*` columns with a positive (delayed) status |
| `max_delay` | Maximum repayment-delay code across the 6 months |
| `avg_repayment_status` | Mean of the 6 `PAY_*` codes |
| `avg_bill` | Mean of the 6 monthly bill amounts |
| `avg_payment` | Mean of the 6 monthly payment amounts |
| `recent_utilisation` | `BILL_AMT1 / LIMIT_BAL` — how much of the credit limit is currently used |

These are aggregates of existing information (no external data), designed
to summarise repayment trend and utilisation rather than relying on any
single monthly snapshot.

## 5. Validation Strategy

5-fold **stratified** cross-validation (`StratifiedKFold`, `shuffle=True`,
`random_state=50`) was used throughout, so the ~22% default rate is
preserved in every fold. All model comparisons and hyperparameter
searches used the same CV split for consistency. For the two boosting
models, out-of-fold (OOF) predictions were also collected so that blend
weights could be optimised without leaking test information.

## 6. Models Tested

| Model | Setup | Mean CV Log Loss |
|---|---|---|
| Logistic Regression | one-hot encoding + standard scaling, `max_iter=1000` | 0.4365 |
| XGBoost (baseline, no engineered features) | `n_estimators=500, lr=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8` | 0.4272 |
| CatBoost (baseline, no engineered features) | `iterations=500, lr=0.03, depth=5`, native categorical handling | 0.4266 |
| XGBoost + engineered features | same hyperparameters as above | 0.4264 |
| CatBoost + engineered features | same hyperparameters as above | 0.4250 |

Both boosting models clearly outperformed logistic regression, and the
engineered features gave a small but consistent improvement for both.
CatBoost's native categorical handling gave it a slight edge over XGBoost
(which required one-hot encoding of `SEX`/`EDUCATION`/`MARRIAGE`).

## 7. Hyperparameter Tuning

A small manual grid search was run over `iterations`/`learning_rate`/`depth`
(CatBoost) and `n_estimators`/`learning_rate`/`max_depth` (XGBoost) using
the same 5-fold CV setup on the feature-engineered data. Best settings
found:

- **CatBoost**: `iterations=700, learning_rate=0.02, depth=6` → CV log loss 0.42435
- **XGBoost**: `n_estimators=500, learning_rate=0.03, max_depth=3` → CV log loss 0.42538

## 8. Ensembling

Out-of-fold predictions from the tuned XGBoost and tuned CatBoost models
were blended with a linear weighted average, and the weight was chosen by
minimising OOF log loss with `scipy.optimize.minimize_scalar`:

- **Optimal weights**: 0.2128 × XGBoost + 0.7872 × CatBoost
- **Blended OOF log loss**: 0.42427

This is a modest improvement over CatBoost alone (0.42435) and a larger
improvement over XGBoost alone (0.42538). An earlier, un-tuned 50/50
blend was also tested and gave 0.42495 — worse than the weight-optimised
version, confirming the optimisation step was worth doing.

No post-processing (e.g. calibration, clipping, or manual threshold
adjustment) was applied to the blended probabilities beyond the linear
combination itself.

## 9. Final Model

Both the tuned XGBoost and tuned CatBoost models were refit on 100% of
the training data (24,000 rows) and used to predict on the 6,000-row test
set. The final submitted probability for each customer is:

```
final_prediction = 0.2128142851 × XGBoost_probability
                  + 0.7871857149 × CatBoost_probability
```

This produced `submission_tuned_ensemble.csv`, the file submitted for
finalist-selection review.

## 10. Key Results

- Best single model: tuned CatBoost, CV log loss 0.42435
- Final blended model: CV/OOF log loss 0.42427
- Most predictive feature: most recent repayment status (`PAY_0`), consistent
  with the EDA finding that repayment delay is the single strongest driver
  of default in this dataset

## 11. Limitations

- **Sample size**: 24,000 training rows is modest for this feature count;
  CV standard deviation across folds was ~0.006, so small log-loss
  differences between model variants (e.g. 0.4250 vs 0.4264) should be
  read as suggestive rather than conclusive.
- **Undocumented categories**: `EDUCATION` and `MARRIAGE` contain category
  codes (`0`, and `5`/`6` for `EDUCATION`) that aren't in the standard
  UCI-style documentation for this kind of dataset. These were left
  as-is; a more careful treatment (e.g. grouping into "other") was
  identified as a possible improvement but not implemented due to time
  constraints.
- **Demographic attributes**: `SEX`, `EDUCATION`, and `MARRIAGE` were
  included as predictors. EDA showed their effect on default is real but
  small relative to repayment behaviour. No fairness/disparate-impact
  analysis across these groups was performed; this is flagged as a
  limitation given the sensitivity of using demographic attributes in a
  credit-risk context, and would be a priority next step if this model
  were used beyond the competition setting.
- **No calibration check**: log loss rewards calibrated probabilities,
  but no explicit calibration curve/reliability diagram was produced to
  verify calibration quality beyond the CV log-loss score itself.
- **Blend simplicity**: only a 2-model linear blend was tried; stacking or
  a 3rd model type was not explored due to time constraints.
