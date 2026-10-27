# Notebook Walkthrough — a cell-by-cell explanatory script

**Predicting Student Failure in Distance Learning Using Machine Learning and Explainability**

This document is a narration of the notebook
`OULAD_student_failure_pipeline.ipynb`. It is written for a reader who knows
data science and machine learning but has never seen this project. For each
block of code it explains *what* the cell does, *why* it is done that way, and
the questions a careful reviewer tends to ask — with answers.

The framing question for the whole project: **can we flag students at risk of
failing a distance-learning course early enough to help them, and can we explain
each flag?** "Early enough" is the constraint that shapes almost every decision
below — it is why we are so careful about *when* information becomes available.

A one-line orientation to the dataset: OULAD (Open University Learning Analytics
Dataset, Kuzilek et al., 2017) is seven linked tables covering student
demographics, course registrations, assessment submissions, and a very large
virtual-learning-environment (VLE) clickstream, across several course modules
and presentations (a "presentation" is one run of a module, e.g. the
February vs October intake).

---

## §0 · Setup

**The cell:** environment flags, imports, a logger, a global seed, and a helper
to show saved figures inline.

What it does. It sets two TensorFlow environment variables *before* anything
imports TensorFlow, imports the scientific stack, builds a single timestamped
logger that writes to stdout, seeds every random number generator, and defines
`show_images()` to render saved PNGs in the notebook.

Why this way.

* **The TF flags come first** because TensorFlow reads `TF_CPP_MIN_LOG_LEVEL`
  and the oneDNN flag only at import time. Setting them after `import tensorflow`
  would be too late — hence they are the very first lines of the notebook.
* **A logger instead of `print`.** We get timestamps, levels, and a single
  consistent stream. Crucially it writes to *stdout*: notebooks render anything
  on stderr (which is where logging goes by default) with a red background that
  looks like an error, which would be misleading for routine progress messages.
* **One global seed** (`set_global_seed`) seeds Python's `random`, NumPy, and
  TensorFlow together. Reproducibility is a stated requirement of the project,
  and the model-selection results depend on stochastic steps (train/test split,
  CV shuffling, SMOTE, random search, network initialisation).
* **`show_images()`** exists because the plotting functions deliberately *save
  figures to disk and close them* (so the run produces a durable artefact set).
  The helper then displays those saved files inline, which keeps the on-disk
  outputs and the on-screen outputs identical.

Reviewer questions.

* *Does seeding guarantee identical results every run?* On CPU, effectively yes
  for the scikit-learn and XGBoost paths. The Keras MLP can still show tiny
  numeric differences from floating-point reduction order and threading; the
  seed removes the large sources of variance (initial weights, data order) but
  is not a bit-for-bit determinism guarantee on all hardware.
* *Why suppress all warnings?* Purely for a clean narrative in a teaching
  notebook; the warnings here are deprecation and convergence chatter that do
  not affect results. Comment that line out to audit them.

---

## §1 · Configuration  *(Phase 1)*

**The cell:** a single `config` namespace holding every tunable parameter.

What it does. It defines one `SimpleNamespace` with paths, the random seed, the
course scope, the target definition, the prediction-cutoff policy, the feature
groups, the imbalance strategies, cross-validation and search settings, the
model hyper-parameter spaces, SHAP settings, and the output sub-directories
(which it creates).

Why this way. Every downstream cell reads from `config`, so there are no magic
numbers scattered through the code and the entire experiment can be reconfigured
in one place — which is also what makes the run reproducible and auditable. A
`SimpleNamespace` (rather than a dict) gives attribute access (`config.CV_FOLDS`)
and easy mutation, which the next cell uses to apply "quick mode".

The choices that matter most live here:

* **Target definition.** `final_result` is collapsed to a binary label:
  Pass/Distinction → 0, Fail → 1. Withdrawals are resolved against a **census
  date**: a student who unregisters before the census never genuinely engaged
  and is dropped; a withdrawal on or after census is treated as failure. This
  follows the at-risk convention in Hlosta et al. (2017) and avoids both
  diluting the "fail" class and rewarding the model for predicting students who
  were never really there.
* **Prediction cutoff.** `CUTOFF_MODE="first_assessment"` sets the decision
  point to the first assessment's due date per presentation. This defines the
  early-warning horizon and is the anchor for all leakage control (see §4).
* **PR-AUC as the primary metric**, because the positive (fail) class is the
  minority and the cost of interest is ranking at-risk students well; ROC-AUC
  and F1 are kept as secondary, comparable references.
* **`A1_DAYS_EARLY_CLIP=(-30, 30)`** caps the submission-timing feature so a few
  extreme outliers cannot dominate a model or its explanation (see §4).

Reviewer questions.

* *Why treat withdrawal-after-census as failure — students withdraw for
  non-academic reasons too?* True, and that is a known limitation; the label is
  a pragmatic proxy used widely in this literature, on the logic that a
  post-census withdrawal is an unsuccessful outcome for the presentation. The
  census split is exactly what stops us from over-claiming here.
* *Why one big config object rather than a dataclass or YAML?* For a
  single-file notebook it is the lightest thing that gives a single source of
  truth and in-notebook mutability; in the packaged version this is a real
  module and could be swapped for a dataclass without touching the logic.

---

## §2 · Data loading & settings  *(Phase 1)*

### Notebook settings (the small cell)

What it does. Exposes the knobs you actually change between runs: `COURSES`,
`QUICK`, the `RUN_EDA / RUN_SHAP / RUN_TRANSFER` toggles, and
`USE_SYNTHETIC_IF_MISSING`. When `QUICK` is on it shrinks CV folds, random-search
iterations, MLP epochs, and SHAP sample sizes.

Why this way. Long runs (full search + the neural net across folds and courses)
are expensive; `QUICK` lets you prove the whole pipeline end-to-end in minutes
before committing to a full run. Keeping these flags separate from `config`
signals "these are the things a user toggles" versus "this is the experiment's
definition".

Reviewer question. *Do quick-mode numbers count?* No — quick mode is for
plumbing and smoke-testing. Reported results should come from `QUICK=False`. The
cell logs which mode is active precisely so a result is never mistaken for a
final one.

### Data loader (the large cell)

What it does. `load_raw()` reads the seven CSVs, validating each against the
documented OULAD column schema; `filter_to_target_modules()` restricts every
table to the courses in scope.

Why this way. The schema check fails fast with a readable message if a file is
malformed or the wrong CSV is supplied — much better than a cryptic error 200
lines later. Filtering to the four Level-1 modules early also matters for
practicality: the VLE clickstream has ~10 million rows, so narrowing scope up
front keeps memory and runtime sane.

Reviewer question. *Is the loader brittle to schema drift?* It is deliberately
strict because OULAD's schema is fixed and published; if a column is missing or
renamed, you want to know immediately rather than silently produce wrong
features.

### Synthetic data generator (testing only)

What it does. `generate_synthetic()` writes tiny CSVs with the exact OULAD schema
and a built-in learnable signal: a latent per-student "ability/engagement"
factor drives click volume, first-assessment behaviour, and the final result.

Why this way. The notebook must run for someone who has not yet downloaded the
real data (and for automated smoke tests). Building a genuine latent signal —
rather than pure noise — means the models can actually learn something, so a
test run exercises the full modelling and explanation path, not just the I/O.

Reviewer question. *Could synthetic results mislead?* They are never presented as
findings; the cell and the markdown both label them testing-only, and the
resolution cell below emits a warning whenever synthetic data is used.

### Resolve the data directory and load the tables

What it does. If `./data` contains all seven CSVs it uses them; otherwise, and if
permitted, it generates synthetic data and proceeds. Then it loads and filters.

Why this way. Graceful degradation: the notebook is runnable out-of-the-box yet
prefers real data whenever present, and it is loud about which one it used.

---

## §3 · Target, census & cutoff  *(Phase 2)*

**The cell:** `build_target`, `compute_cutoffs`, `build_base_table`, and
per-field missing-value handling.

What it does. It constructs the binary target with the census-aware withdrawal
logic from §1, computes each presentation's prediction cutoff as the earliest
assessment due date (plus any configured offset), assembles the student-level
base table, and imputes missing values per field (mode for `imd_band`, median for
`date_registration`).

Why this way.

* **Per-presentation cutoffs**, not one global day, because different
  presentations have different assessment calendars; anchoring on each
  presentation's own first assessment keeps the early-warning horizon
  comparable across cohorts.
* **Per-field imputation.** `imd_band` (a deprivation decile) is categorical, so
  the mode is the natural fill; `date_registration` is numeric and skewed, so the
  median is robust. A blanket imputation would be wrong for at least one of them.

Reviewer questions.

* *Why drop pre-census withdrawals rather than label them?* Because they
  represent students who never genuinely started the presentation; including
  them would pollute the "fail" class with non-engagement that the early-warning
  features (which only exist once a student is active) cannot speak to.
* *Is median imputation of a date defensible?* It is a pragmatic choice for a
  small number of missing registration offsets; the field is also available to
  the model as-is, and the fraction imputed is reported in the EDA missingness
  table so the reader can judge its impact.

---

## §4 · Feature engineering  *(Phase 2)*

**The defining cell** builds three feature families and a feature registry; **the
short cell after it** runs them, writes `feature_matrix.csv`, and splits the
matrix per course.

What it does. From the clickstream it computes VLE-engagement features (total and
recent clicks, active days, clicks per active day, longest consecutive active
streak, days since last login, first-week clicks, pre-start clicks, an engagement
trend slope, and site/activity diversity) — **all restricted to dates on or
before the cutoff**. From the first assessment (A1) it computes whether the
student submitted, the score, a banked flag, and submission timing
(`a1_days_early`, clipped, with a late flag from the uncapped sign). Students with
no clicks before the cutoff are encoded as zero-click rather than missing.
`feature_columns()` then records which columns are categorical vs numeric so the
modelling and explanation stages stay in sync with the matrix.

Why this way. The behavioural features are designed to capture *disengagement*,
which is the leading signal for distance-learning failure: not just how much a
student clicked, but whether their activity is trailing off (`engagement trend`),
how long since they last appeared (`days since last login`), and whether they
got going at all in week one. The strict clip to the cutoff is what makes these
honest early-warning features rather than hindsight.

Reviewer questions.

* *The big one: isn't using the A1 score leakage?* This is the right question to
  press on. The cutoff is the first assessment, so the prediction point is best
  read as **"just after the first assessment result is known."** That is a
  deliberate, literature-standard early-warning checkpoint (Hlosta et al. predict
  at the first-assessment milestone): in practice an instructor intervenes after
  the first piece of graded work. The VLE features are strictly *before* that
  point; the A1 features are the *event that defines* the checkpoint. If you want
  a stricter, pre-A1 horizon, the design supports it without code changes — set
  `CUTOFF_MODE="fixed_day"` (predict at, say, day 30 regardless of assessments)
  or add a positive `CUTOFF_ASSESSMENT_OFFSET`. The point is that the horizon is
  an explicit, configurable choice, not an accident.
* *Why clip `a1_days_early`?* A handful of students submit extremely late (a
  resubmission months on), producing values dozens of standard deviations out
  that would otherwise dominate both a linear model's coefficient effect and the
  SHAP attribution for a single odd case. Clipping to ±30 days bounds the
  magnitude; the late/early flag is taken from the *uncapped* sign so a genuinely
  late submission still reads as late. It is configurable (`None` disables it).
* *Why zero-click instead of missing for absent VLE?* Because "no clicks before
  the cutoff" is information, not absence of information — it is a strong
  disengagement signal, and encoding it as zero lets the model use it.

---

## §5 · Exploratory data analysis  *(Phase 2)*

**The defining cell** holds the EDA functions; **the runner cell** executes them
and displays the figures inline.

What it does. It produces the overall class balance, fail rate by course, fail
rate by each of five demographics (gender, age band, disability, highest
education, IMD band) as both CSVs and bar charts, the distributions of key
engagement/assessment features split by outcome, and a numeric-feature
correlation heatmap. Residual missingness is also tabulated.

Why this way. EDA here is not decorative — each artefact justifies a later
modelling decision:

* The **class-balance** figure quantifies the imbalance, which is what motivates
  PR-AUC as the primary metric and the resampling/weighting comparison in §6.
* The **demographic fail-rate** breakdowns serve two purposes: a fairness lens
  (are outcomes uneven across protected-ish attributes?) and a feature-usefulness
  check (do these attributes carry signal at all?).
* The **correlation heatmap** is explicitly there to monitor the SHAP
  correlation caveat: when features are correlated, attribution can be split or
  shifted between them, so we want to see the correlation structure before
  interpreting SHAP.
* In the distribution plots, click counts are clipped at the 99th percentile *for
  the visualisation only*, so heavy tails do not flatten the curves.

Reviewer question. *Why surface demographic disparities — are they used by the
model?* The demographic features are available to the models (they can carry
signal), but the breakdowns are shown so disparities are visible and can be
discussed rather than hidden; how interventions are then targeted is a policy
decision the analysis should inform responsibly, not a licence to act on a
protected attribute mechanically.

---

## §6 · Class-imbalance strategies  *(Phase 2)*

**The cell:** `_build_pipeline` and `select_strategy`.

What it does. For a given model it builds an imbalanced-learn pipeline
(`preprocess → [SMOTE] → classifier`) and, via `select_strategy`, compares three
ways of handling imbalance — none, SMOTE, and cost-sensitive class weighting — by
stratified cross-validated PR-AUC, returning the best. If every strategy errors
(e.g. a fold too small to resample) it falls back to a safe default.

Why this way.

* **SMOTE lives inside the pipeline**, so it is applied *within each CV fold's
  training split only*. Resampling before cross-validation is a classic leakage
  mistake that inflates scores; putting SMOTE in the pipeline guarantees the
  validation fold is never synthesised from or contaminated by resampling.
* **Comparing strategies per model** rather than fixing one, because the best
  imbalance treatment is data- and model-dependent — tree ensembles often do
  fine with class weights, while a linear or neural model may benefit more from
  SMOTE, and sometimes nothing beats leaving it alone.
* **Selection metric = average precision = PR-AUC**, matching the primary metric
  so we select strategies for the objective we actually care about.

Reviewer questions.

* *Does SMOTE make sense on one-hot-encoded data?* SMOTE interpolates in the
  transformed numeric space here. It is a reasonable default and is applied after
  encoding/scaling; if one wanted to respect categorical structure more strictly,
  SMOTE-NC would be the alternative. The strategy comparison guards against
  picking SMOTE when it does not actually help.
* *Why not just always use class weights?* Because the comparison sometimes finds
  SMOTE or "none" gives better cross-validated PR-AUC; we let the data decide per
  model and report which strategy won.

---

## §7 · Models  *(Phase 3)*

**The cell:** the preprocessing transformer, the Keras MLP wrapper, and the model
factory.

What it does. `_SanitizingPreprocessor` one-hot-encodes categoricals, scales
numerics (optional), and **sanitises feature names**; `build_preprocessor`
exposes it. `KerasMLPClassifier` wraps a dropout MLP in the scikit-learn
estimator API. `get_classifier` is a factory returning Logistic Regression,
Random Forest, XGBoost, or the MLP, and helper tables record which models scale,
which support class weights, and XGBoost's `scale_pos_weight`.

Why this way.

* **Name sanitisation.** One-hot encoding OULAD produces feature names such as
  `age_band=55<=` and `a1_score_band_fail_<40`. XGBoost rejects the characters
  `[`, `]`, `<`. Sanitising once, immediately after encoding, means the models,
  the persisted pipelines, and the SHAP explainers all see the same clean,
  unique names — so explanations are labelled consistently with the model input.
* **A custom sklearn-compatible Keras wrapper** lets the neural network ride the
  *same* cross-validation, tuning, and SHAP machinery as the other three models,
  so comparisons are apples-to-apples. Three details earn their keep: the class
  declares `ClassifierMixin` before `BaseEstimator` so scikit-learn's tag system
  recognises it as a classifier (otherwise probability scoring inside CV breaks);
  it clears the Keras session on each build to stop TensorFlow memory growing
  across the many fits; and it supports early stopping and class weights.
* **Tree models skip scaling** (it is unnecessary for them), while LR and the MLP
  are scaled — handled by the `scale` flag rather than two separate code paths.
* **XGBoost uses `scale_pos_weight`** for its cost-sensitive option, which is the
  idiomatic XGBoost lever, rather than a generic `class_weight`.

Reviewer questions.

* *Why hand-roll the Keras wrapper instead of using scikeras?* To keep
  dependencies minimal and retain full control over the build, early stopping,
  session clearing, and the classifier tags — and to make the SHAP integration
  explicit. scikeras would also work.
* *Why sanitise names instead of stripping the offending characters at the
  source?* Because we want to keep human-readable, faithful category names for
  interpretation; sanitising in one place keeps the mapping consistent everywhere
  downstream instead of mutating the raw data.

---

## §8 · Training & evaluation  *(Phase 3)*

**Three cells:** the metrics/plots, the `train_course` routine, and the loop that
runs it per course and shows the leaderboard.

What it does. `compute_metrics` returns PR-AUC, ROC-AUC, F1, precision, recall and
the confusion counts at the decision threshold; `plot_curves` and
`plot_confusion` save the PR/ROC overlays and confusion matrix. `train_course`
performs a stratified train/test split, selects the imbalance strategy (§6),
tunes hyper-parameters, refits the best configuration, evaluates on the held-out
test set, and persists the fitted pipeline plus the transformed train/test
matrices (with sanitised names) that SHAP will need. The final cell runs this for
each course and displays a per-course metrics table and curves.

Why this way.

* **A held-out test set on top of cross-validation.** CV is used for model
  selection and tuning; the untouched test split gives an unbiased final estimate
  of generalisation. Reporting only CV scores from the same data used to choose
  the model would be optimistic.
* **Grid search for Logistic Regression, randomised search for RF/XGBoost.** LR
  has a tiny, low-dimensional grid where exhaustive search is cheap; the tree
  models have large spaces where randomised search finds strong configurations
  far more efficiently than a full grid.
* **Persisting the transformed matrices** matters because SHAP must operate in
  the model's input space with the exact (sanitised) feature names; recomputing
  them later risks subtle drift, so we store them with the model.

Reviewer questions.

* *Why PR-AUC over ROC-AUC as the headline?* Under class imbalance ROC-AUC can
  look flattering because true negatives dominate; PR-AUC focuses on the
  minority (fail) class we actually care about ranking well. We still report
  ROC-AUC for comparability.
* *Why a fixed 0.5 threshold for F1/confusion rather than a tuned one?* The
  primary metrics (PR-AUC, ROC-AUC) are threshold-free, which is what we select
  on; the confusion view at 0.5 is a readable operating-point snapshot, and the
  threshold is exposed in config for anyone who wants to tune it to an
  intervention budget.

---

## §9 · Explainability with SHAP  *(Phase 4)*

**The defining cell** holds the SHAP routines; **the runner cell** explains each
course's best model and shows the plots.

What it does. `compute_shap` picks the right explainer per model: the fast exact
`TreeExplainer` for Random Forest and XGBoost (with a fallback to a sampled
model-agnostic explainer if it cannot read the model), `LinearExplainer` for
Logistic Regression, and a sampled `KernelExplainer` on `predict_proba` for the
MLP. It normalises the output to the positive class, then produces a global
beeswarm plot, a mean-|SHAP| bar chart with an importance CSV, and per-student
**waterfall** plots for representative true-positive, false-positive and
false-negative cases.

Why this way.

* **Explainer matched to model** for both correctness and speed: tree explainers
  are exact and fast on tree ensembles; the linear explainer is exact for LR; the
  MLP has no special-case explainer, so a sampled model-agnostic one is used.
* **The TreeExplainer fallback** exists because some SHAP/XGBoost version
  combinations serialise the model's `base_score` in a form older SHAP builds
  cannot parse; rather than let the whole explanation step fail, we drop to a
  model-agnostic explainer so SHAP always returns. (Upgrading SHAP restores the
  fast exact path.)
* **TP/FP/FN waterfalls** turn SHAP into error analysis: they show not just which
  features matter globally but *why a specific student was flagged*, and let you
  inspect what the model got wrong and which features drove the mistake.

Reviewer questions.

* *How are correlated features handled in the attributions?* This is the SHAP
  correlation caveat, and it is exactly why §5 shows the correlation heatmap —
  attributions among correlated features should be read as a group. The tree
  explainer here uses the path-dependent perturbation, so attributions reflect
  the model's structure; we flag the caveat rather than over-interpret individual
  correlated features.
* *Isn't KernelExplainer slow?* Yes on full data, which is why the MLP and the
  fallback path use a sampled background and a sample of rows (sizes set in
  config) — enough for stable global rankings and representative local plots
  without an impractical runtime.

---

## §10 · Cross-course generalisation  *(Phase 5)*

**The defining cell** holds the cross-course analyses; **the runner cell**
executes them when more than one course is in scope.

What it does. `summarise_within_course` builds a model×course PR-AUC heatmap;
`transfer_matrix` trains XGBoost on one course and tests it on the others,
producing a train→test PR-AUC matrix; `shap_consistency` checks how often each
feature appears in the top-k most important features across courses.

Why this way. Per-course accuracy is not the same as a *generalisable*
early-warning system. The transfer matrix asks the sharper question: does a model
fitted on one cohort still rank at-risk students in a different course? Strong
off-diagonal PR-AUC means the disengagement signals are portable; weak
off-diagonal means each course needs its own model. The SHAP-consistency check
complements this by asking whether the *drivers* are the same across courses —
construct validity for the story we tell about why students fail.

Reviewer questions.

* *Why use XGBoost specifically for the transfer matrix?* It is the project's
  primary model; using a single consistent learner makes the off-diagonal numbers
  comparable and isolates the effect of changing the train/test course from the
  effect of changing the model.
* *How should the matrix be read?* Diagonal = train and test on the same course
  (an upper reference); off-diagonal = transfer. The gap between them is the
  generalisation cost of moving to an unseen course.

---

## §11 · Summary  *(Phase 5)*

**The cell:** writes `run_summary.json` and prints the best model per course.

What it does. It serialises the consolidated results (best model and its metrics
per course) to JSON and logs a compact leaderboard.

Why this way. A single machine-readable summary is the hand-off artefact for the
write-up: it captures, in one place, which model won each course and at what
PR-AUC, so the report can cite results without re-running anything.

---

## Closing notes for a reviewer

* **Reproducibility.** One seed flows through every stochastic step; splits and
  CV are stratified and seeded; the full configuration is captured in `config`
  and the results in `run_summary.json`.
* **The horizon is the heart of it.** Almost every design decision — the census
  split, the per-presentation cutoff, the strict pre-cutoff clipping of VLE
  features — exists to keep the predictions honestly *early*. The first-assessment
  checkpoint is an explicit, configurable choice, and the most important thing to
  agree on before trusting the numbers.
* **Known limitations worth stating.** The fail label folds in post-census
  withdrawals (a proxy, not ground truth about academic ability); SMOTE operates
  in encoded space; SHAP attributions should be read with the correlation
  structure in mind; and quick-mode numbers are for plumbing, not reporting.
