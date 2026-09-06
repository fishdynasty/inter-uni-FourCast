# README — Reproduction Instructions

## What this reproduces

Running `code/full_pipeline.py` regenerates `submission_tuned_ensemble.csv`,
the exact file being submitted for finalist-selection review.

## Folder structure expected

Place the files like this:

project/
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   ├── Data_clean.ipynb
│   ├── EDA.ipynb
│   └── Modelling.ipynb
├── code/
│   └── full_pipeline.py
├── report/
│   └── Final_Model_Info.md
├── submission/
│   └── submission_tuned_ensemble.csv
├── README.md
├── DISCLOSURE.md
└── requirements.txt

If `full_pipeline.py` is run from inside the `code/` folder that sits
alongside `data/`, `report/`, and `submission/` as shown above, the
relative paths in the script will work unmodified.

If a different folder structure is used, update the relevant path
strings inside `full_pipeline.py`.

## Required files

- `train.csv` — 24,000 rows, competition-provided training dataset
- `test.csv` — 6,000 rows, competition-provided test dataset

No external datasets or pretrained model weights are required.

See `DISCLOSURE.md` for details regarding external tools, code,
packages, and AI assistance used during development.

## Key dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- catboost
- scipy

Exact package versions were not pinned during initial development.

Before final submission, replace the placeholder `requirements.txt`
with the package versions from the environment used to generate the
final leaderboard submission.

This can be generated using:

pip freeze > requirements.txt

## Random seeds

`random_state` / `random_seed` = **50** for all stochastic modelling
and validation steps where a seed is applicable.

This includes:

- internal train-validation split
- `StratifiedKFold`
- `XGBClassifier`
- `CatBoostClassifier`

The same seed is used consistently across `Modelling.ipynb` and
`full_pipeline.py`.

## Execution order

### 1. Data cleaning notebook — optional for reproduction

Run:

`notebooks/Data_clean.ipynb`

This notebook performs data quality checks and documents the cleaning
process.

No transformed dataset is produced from this notebook, and the final
prediction pipeline does not depend on running it.

### 2. Exploratory data analysis notebook — optional for reproduction

Run:

`notebooks/EDA.ipynb`

This notebook contains exploratory analysis and visualisations used to
better understand the dataset.

It is not required to reproduce the final submission file.

### 3. Final modelling pipeline — required

Run:

`code/full_pipeline.py`

This script performs the complete final pipeline:

1. Loads the raw training and test datasets.
2. Performs the required preprocessing.
3. Generates engineered features.
4. Trains the tuned XGBoost model.
5. Trains the tuned CatBoost model.
6. Generates default probabilities for the test set.
7. Blends the XGBoost and CatBoost predictions using the fixed weights
   documented in `report/Final_Model_Info.md`.
8. Writes the final prediction file to
   `submission/submission_tuned_ensemble.csv`.

To run the pipeline:

cd code
python full_pipeline.py

`full_pipeline.py` is the canonical script used for reproduction.

`Modelling.ipynb` contains the modelling process interactively,
including model comparison, cross-validation, and hyperparameter
experiments that motivated the final model settings.

## Final model / ensemble

The final leaderboard submission uses an ensemble of:

- Tuned XGBoost
- Tuned CatBoost

The final submission probabilities are produced by blending the
predictions from these two models.

Full details, including model hyperparameters, ensemble weights, and
other modelling settings, are documented in:

`report/Final_Model_Info.md`

## Validation strategy

Model performance was evaluated using stratified cross-validation.

`StratifiedKFold` was used so that each validation fold maintained
approximately the same proportion of default and non-default customers
as the full training dataset.

This provides a more reliable estimate of model performance than relying
on a single random train-validation split.

The primary evaluation metric was log loss because the competition
requires predicted default probabilities rather than only binary
classifications.

Lower log loss indicates better probabilistic prediction performance,
while highly confident incorrect predictions receive a larger penalty.

The main model comparison and cross-validation experiments are recorded
in:

`notebooks/Modelling.ipynb`

The final model settings used in `full_pipeline.py` were selected based
on these validation experiments.

## Models tested

The modelling process compared multiple approaches, including:

- Logistic Regression
- XGBoost
- CatBoost

Logistic Regression was used as a baseline model.

XGBoost and CatBoost were then tested to capture more complex,
non-linear relationships between customer characteristics, repayment
behaviour, and default risk.

The final approach uses tuned XGBoost and CatBoost models combined in an
ensemble.

## Feature engineering

The final pipeline includes engineered features derived from the
competition-provided variables.

These features are generated automatically inside `full_pipeline.py`
before model training.

No external data is required for feature generation.

The feature engineering process is fully reproducible from the original
`train.csv` and `test.csv` files.

## Ensembling and post-processing

The final predictions are produced by blending the predicted default
probabilities from the tuned XGBoost and tuned CatBoost models.

The ensemble uses fixed model weights documented in:

`report/Final_Model_Info.md`

No manual modification of individual customer predictions is performed
after the ensemble probabilities are generated.

The blended probabilities are written directly to the final submission
file.

## Final submission file

The exact prediction file being submitted for finalist-selection review
is:

`submission/submission_tuned_ensemble.csv`

This file is generated directly by:

`code/full_pipeline.py`

The file should not be manually modified after generation.

## Validation and leaderboard results

The final ensemble was selected based on local stratified
cross-validation performance.

- Final local CV score: **[INSERT FINAL CV SCORE]**
- Final leaderboard score: **[INSERT FINAL LEADERBOARD SCORE]**
- Final submission file: `submission_tuned_ensemble.csv`

Detailed experiment and model comparison results are provided in
`notebooks/Modelling.ipynb` and the methodology report.

## Notes on other files in this package

### `submission_catboost.csv`

This file represents an earlier single-model CatBoost approach developed
before feature engineering, tuning, and ensemble blending were
finalised.

It is retained for completeness and audit history.

It is **not** the final submission being reviewed.

### `Submission2_Remainnig_code`

This contains code associated with the earlier CatBoost-only submission.

It is retained for transparency but is not required to reproduce the
final leaderboard submission.

### `submission.csv`

This is an earlier exploratory prediction file.

It is also **not** the final submission under review.

The final file is:

`submission_tuned_ensemble.csv`

## Disclosure

Full disclosure information is provided in:

`DISCLOSURE.md`

This includes information regarding:

- external datasets used;
- external code, notebooks, repositories, or public solutions consulted;
- pretrained models;
- AI tools or coding agents;
- manual prediction modification or post-processing;
- additional information used beyond the competition-provided files.

## Limitations and reproducibility notes

The final solution was developed using selected hyperparameter settings
rather than an exhaustive search of every possible model configuration.

Cross-validation provides an estimate of generalisation performance,
but performance on the hidden competition test set may differ.

Exact dependency versions should therefore be preserved in
`requirements.txt` to improve reproducibility.

## Reproduction summary

To reproduce the final leaderboard submission:

1. Place `train.csv` and `test.csv` inside `data/raw/`.
2. Install the packages listed in `requirements.txt`.
3. Navigate to the `code/` folder.
4. Run `python full_pipeline.py`.
5. The required engineered features are generated automatically.
6. The tuned XGBoost and CatBoost models are trained.
7. Test-set default probabilities are generated.
8. The model probabilities are blended using the documented ensemble
   weights.
9. `submission/submission_tuned_ensemble.csv` is generated.

The resulting file should correspond to the team's final leaderboard
submission being reviewed.

