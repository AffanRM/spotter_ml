# Manual Git checkpoints

Commits and pushes are performed by the candidate. These commands assume
PowerShell or Git Bash/MSYS2 is open in the project folder. The first two
checkpoints were committed and pushed to `https://github.com/AffanRM/spotter_ml.git`.
Checkpoint 3 was committed and pushed as `af62a46`, and checkpoint 4 as `cabec32`.
Checkpoint 5 was committed and pushed as `b70e837`, and checkpoint 6 as `14207a8`.
Start with checkpoint 7
for the new changes. Earlier instructions are retained
for reference.

## First-time initialization

Create an empty GitHub repository (without a generated README or gitignore).
Then run:

```powershell
git init
git branch -M main
```

## Checkpoint 1: assessment inputs and isolated environment setup

```powershell
git add .gitignore SETUP.md TASK_OVERVIEW.md requirements.txt requirements-project.txt requirements-lock.txt score.py freight-rate-ml-assessment-1-.pdf train-test.csv validation.csv validation-predictions-template.csv december-chart-inputs.csv
git diff --cached --stat
git commit -m "chore: add assessment inputs and reproducible Python environment"
```

The raw CSVs and PDF are deliberately staged by name here so a clone contains
the assessment inputs needed to reproduce this solution. The virtual environment,
temporary files, and Python caches are excluded by `.gitignore`.

## Checkpoint 2: reproducible exploration and validation foundations

Run the analysis and tests first:

```powershell
.\.venv\Scripts\python.exe -m freight.explore
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git add readme.md GIT_WORKFLOW.md freight tests reports/eda
git diff --cached --stat
git diff --cached --check
git commit -m "feat: add reproducible data audit and temporal validation utilities"
```

## Connect and push

The existing remote is already configured. For a fresh repository only:

```powershell
git remote add origin https://github.com/AffanRM/spotter_ml.git
git push -u origin main
```

For later checkpoints, review and stage the specific changed files, commit,
then run `git push`. Use `git status` to see what is staged or still untracked.
Do not run initialization or `git remote add` again once they have succeeded.

## Checkpoint 3: development model comparisons

With the virtual environment active, these commands work in either shell:

```text
python -m unittest discover -s tests -v
python -m freight.benchmark
git status
git add .gitignore SETUP.md TASK_OVERVIEW.md readme.md GIT_WORKFLOW.md
git add freight/features.py freight/models.py freight/benchmark.py tests/test_models.py reports/benchmark
git diff --cached --stat
git diff --cached --check
git commit -m "feat: benchmark baseline and CatBoost models on temporal backtests"
git push
git status
```

This checkpoint keeps large generated prediction diagnostics in ignored
`artifacts/`. Summary metrics, the experiment manifest, and the comparison report
are committed. Fixed seeds and parameters make metrics reproducible with the
locked environment; recorded runtimes vary by machine. September-October remains
reserved for the later final check. No dependency additions are needed.

## Checkpoint 4: unseen-city stress tests and error analysis

With the virtual environment active:

```text
python -m unittest discover -s tests -v
python -m freight.robustness
git status
git add freight/splits.py freight/robustness.py tests/test_robustness.py reports/robustness
git add readme.md SETUP.md TASK_OVERVIEW.md GIT_WORKFLOW.md
git diff --cached --stat
git diff --cached --check
git commit -m "feat: add unseen-city stress tests and residual diagnostics"
git push
git status
```

If running from a fresh clone, first run `python -m freight.benchmark` to
regenerate the ignored row-level predictions required by robustness. When those
predictions already exist, there is no need to rerun the benchmark for this
checkpoint. The tests now include city isolation in both route directions,
target-independent city selection, and rejection of invalid development
prediction provenance. The September-October final check is still unscored.

## Checkpoint 5: frozen model policy and reserved-window evaluation

The choices in `MODEL_DECISIONS.md` and `model_policy.json` were written before
the reserved-window results were computed. The main model remains the full
CatBoost MAE model, with a separate chart-only model for the reduced inputs.
Scores from this window are reported without further model selection.

```text
python -m unittest discover -s tests -v
python -m freight.final_check
git status
git add .gitattributes model_policy.json MODEL_DECISIONS.md
git add freight/policy.py freight/final_check.py tests/test_policy.py reports/final_check
git add readme.md SETUP.md TASK_OVERVIEW.md GIT_WORKFLOW.md
git diff --cached --stat
git diff --cached --check
git commit -m "feat: freeze model policy and evaluate reserved temporal holdout"
git push
git status
```

The tests now cover invalid policy rejection, model-role contracts, and
protection against overwriting prior final-check evidence with another
experiment. Full-data refits and submission CSV generation come next.

## Checkpoint 6: full-data refits and validated submission files

```text
python -m unittest discover -s tests -v
python -m freight.submission
python -m freight.submission --predict-only
git status
git add freight/submission.py tests/test_submission.py
git add validation_predictions.csv december_predictions.csv scorer_results/candidate_december.png reports/submission
git add readme.md SETUP.md TASK_OVERVIEW.md GIT_WORKFLOW.md
git diff --cached --stat
git diff --cached --check
git commit -m "feat: generate validated submission predictions from full-data models"
git push
git status
```

The first submission command refits the unchanged selected models on 48,000
labeled rows. The second reuses saved models and should reproduce the same CSVs
and chart. Neither command changes the raw inputs or the supplied scorer.
Models and large diagnostics remain excluded from Git under `artifacts/`.
The formal report and Loom recording remain to be prepared.

## Checkpoint 7: full audit and reviewer reproducibility

```text
python -m unittest discover -s tests -v
python -m freight.submission
python -m freight.audit --require-saved-models
git status
git add .github/workflows/checks.yml freight/audit.py freight/submission.py tests/test_submission.py
git add PROJECT_REVIEW.md readme.md SETUP.md TASK_OVERVIEW.md GIT_WORKFLOW.md
git add reports/audit reports/submission
git diff --cached --stat
git diff --cached --check
git commit -m "fix: improve reviewer portability and audit submission integrity"
git push
git status
```

After pushing, inspect the new Project checks workflow in GitHub's Actions tab.
Its Windows/Linux hosted results are pending until that run completes. The audit
preserves all frozen model choices and submission prediction values; regenerated
model metadata and audit manifests can change. The clean-environment evidence
records the audit session's reproduction of commit `14207a8` outputs.
