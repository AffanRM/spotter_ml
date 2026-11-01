# Loom recording script and walkthrough plan

Target: approximately **2 minutes 40 seconds**, including navigation. The
assessment requires **2-3 minutes with camera on**. The script is a rehearsal
aid; speak naturally and only make claims you can explain.

## Prepare before recording

1. Open the repository in your editor. Pre-open the tabs listed below and set
   a readable font size; do not spend recording time searching for files.
2. Open the formal PDF and the scorer chart. Put the camera bubble where it
   does not cover code, figures, or results. Keep the camera on throughout.
3. Activate the environment and run `python -m freight.audit
   --require-saved-models`. Keep its successful output available. Full training
   is already reproducible; do not wait for a refit during a 3-minute recording.
4. Do one timed rehearsal. Aim for 2:35-2:50 with short pauses between sections.

## Timed script

Only the blockquotes are spoken. Screen directions are not part of the script.

### 0:00-0:15 - Objective and evidence

**Show:** README overview/results, then the formal report's first page.

> Hi, I'm Affan. This project predicts freight rates from forty-eight thousand labeled loads. I'll show the data issues, validation approach, model choice, and key code.

### 0:15-0:40 - Exploration and data quality

**Show:** report page 1, then `freight/data.py`, `clean_features` at line 60.

> I found missing weights and market indices, negative weights, and eight cities absent from training. I treat invalid weights as unknown and preserve quality indicators, rather than guessing their intended values. I kept extreme rate labels. The quote signal also changes its relationship with rates across months, so I tested models with and without it.

### 0:40-1:05 - Training and validation

**Show:** `freight/splits.py`, `BACKTESTS` and `temporal_split` at line 39;
report page 2 has the matching split table.

> Because the final loads occur after the training period, I used two expanding chronological backtests with two-month validation windows. I reserved September and October for a final performance check and fixed model choices beforehand. Preprocessing fits on training rows only. I also removed eight cities from training to test generalization to unfamiliar locations.

### 1:05-1:30 - Model reasoning and results

**Show:** report pages 3-4, with the candidate table and reserved metrics.

> I compared eight candidates, including simple baselines, Ridge, and CatBoost variants. Full-feature CatBoost with an MAE objective had the best development MAE. On the reserved window, it achieved one hundred sixteen dollars MAE, versus two hundred twenty-nine for the equipment baseline, a forty-nine percent reduction. Large errors still dominate RMSE, and remain included.

### 1:30-2:05 - Important code and inference paths

**Show:** `model_policy.json`; `freight/features.py`, `build_features` at line 13;
then `freight/submission.py`, `validation_predictions` at line 46.

> The policy file freezes the main and chart models. The feature builder excludes IDs and targets, and gives the chart model only inputs available in the December file. The submission code joins predictions by load ID, preserving template order. It saves and reloads native models, verifies their predictions, and runs the supplied scorer. These checks prevent silent row misalignment or unsupported chart inputs.

### 2:05-2:40 - Deliverables, chart and limits

**Show:** final CSV header, `scorer_results/candidate_december.png`, then the
successful audit output. Finish on the chart or README results.

> Both frozen models are finally refitted on all labeled loads. The scorer validates all twelve thousand final predictions and thirty-one December rows. This chart changes only the date, without inventing unavailable future signals. The repository includes pinned dependencies, tests, provenance checks, and the formal report. Hidden-set accuracy is unknown, and the December curve is a model prediction, not a calibrated holiday forecast.

## Code navigation reference

| File / location | Point to explain |
| --- | --- |
| `freight/data.py:60` | Unknown invalid weights and separate quality flags; no learned statistics |
| `freight/splits.py:39` | Disjoint chronological partitions and inclusive calendar cutoffs |
| `model_policy.json` | Fixed roles, iteration count, seed and evaluation boundaries |
| `freight/features.py:13` | Main versus chart feature availability; no ID/target features |
| `freight/submission.py:46` | ID-based alignment, preserving template order |
| `freight/submission.py:77` and `:98` | Native persistence and checked reloads, if asked |

The line numbers were checked when this walkthrough was prepared. Use function
names if your editor's line numbers later change. Do not attempt to show every
file in the recording; the report and repository provide the full detail.

## If rehearsal exceeds three minutes

Keep the validation explanation and the ID-alignment example. Shorten navigation
and use the report instead of scrolling through each implementation. Keep at
least one actual code example on screen: a report-only presentation does not
meet the important-code walkthrough requirement.

## Final recording check

- Runtime is 2-3 minutes and the camera stays on.
- Audio and text are legible; you explain each code example in your own words.
- Coverage includes findings, quality issues and handling, model reasoning,
  training/split approach, and important code.
- Claims distinguish development/reserved results from unknown hidden scores.
- The Loom link opens for the reviewer without requesting access. Test it in
  a signed-out/private browser window before submitting.

Once recorded, add the actual accessible Loom URL to the delivery checklist.
No recording or Loom link has been created by this script document.
