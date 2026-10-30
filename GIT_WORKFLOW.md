# Manual Git checkpoints

Commits and pushes are performed by the candidate. These commands assume
PowerShell is open in the project folder. No Git repository existed when the
analysis checkpoint was prepared.

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

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with the empty repository you created:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

For later checkpoints, review and stage the specific changed files, commit,
then run `git push`. Use `git status` to see what is staged or still untracked.
Do not run initialization or `git remote add` again once they have succeeded.
