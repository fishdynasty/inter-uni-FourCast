# Required Disclosure

> **Please review and confirm/edit before submitting.** I've drafted this
> from what's verifiable in the files provided, but I don't have full
> visibility into exactly who wrote which parts of these three notebooks
> (you, your teammate, or with AI assistance) — please correct anything
> below that isn't accurate.

**External datasets used:** None. Only the competition-provided
`train.csv` and `test.csv` were used.

**External code, notebooks, repositories, or public solutions consulted
or adapted:** None identified in the provided files. If you or your
teammate referenced any public notebook, Kaggle kernel, blog post, or
Stack Overflow snippet while building this (e.g. for the feature
engineering ideas or the blend-weight optimisation approach), list it
here.

**Pretrained models used:** None. Both the XGBoost and CatBoost models
were trained from scratch on the competition training data.

**AI tools or coding agents used:** _[Please fill in.]_ If any AI
assistant (e.g. Claude, ChatGPT, Copilot) was used to help write, debug,
or explain any part of this code, note it here, including roughly which
parts (e.g. "AI assistance was used for drafting the feature engineering
function and the ensemble-weight optimisation code; model selection and
final hyperparameters were chosen by the team based on the CV results").
Use of AI tools is not automatically considered misconduct — the
requirements document only asks for disclosure of provenance.

**Manual modification or post-processing of predictions:** None. The
submitted probabilities are the direct output of the fixed weighted
blend (0.2128 × XGBoost + 0.7872 × CatBoost) described in
`Final_Model_Info.md`, with no manual editing, clipping, or thresholding
applied afterwards.

**Additional information used beyond the competition-provided files:**
None identified — all feature engineering derives from columns already
present in `train.csv`/`test.csv`.
