# Manual Git checkpoints

Commits and pushes are performed by the candidate. These commands assume
PowerShell or Git Bash/MSYS2 is open in the project folder. The first two
checkpoints were committed and pushed to `https://github.com/AffanRM/spotter_ml.git`.
Checkpoint 3 was also committed and pushed as `af62a46`. Start with checkpoint 4
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
