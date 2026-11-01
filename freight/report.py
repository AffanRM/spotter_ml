"""Build the formal assessment report from audited metrics and the scorer chart."""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from freight.policy import ROOT
from freight.submission import sha256


TEAL = colors.HexColor("#075563")
INK = colors.HexColor("#162C34")
MUTED = colors.HexColor("#53666D")
PALE = colors.HexColor("#EEF4F5")
LINE = colors.HexColor("#D2DFE2")
WIDTH = A4[0] - 92


def build(output: Path):
    audit = json.loads((ROOT / "reports/audit/checks.json").read_text())
    if not audit["technical_checks_passed"]:
        raise ValueError("Run a passing project audit before generating the report")
    eda = json.loads((ROOT / "reports/eda/audit.json").read_text())
    submission = json.loads((ROOT / "reports/submission/manifest.json").read_text())
    final = pd.read_csv(ROOT / "reports/final_check/metrics.csv").set_index("role")
    leaderboard = pd.read_csv(ROOT / "reports/benchmark/leaderboard.csv")
    geographic = pd.read_csv(ROOT / "reports/robustness/location_pooled_metrics.csv")
    residuals = pd.read_csv(ROOT / "reports/robustness/residual_summary.csv").set_index("model")
    chart_path = ROOT / "scorer_results/candidate_december.png"
    if sha256(chart_path) != submission["output_sha256"]["scorer_results/candidate_december.png"]:
        raise ValueError("Scorer chart does not match the audited artifact")
    improvement = 100 * (1 - final.loc["main", "mae"] / final.loc["reference", "mae"])
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleCustom", fontName="Helvetica-Bold", fontSize=27, leading=31, textColor=INK, spaceAfter=13))
    styles.add(ParagraphStyle("Deck", fontName="Helvetica", fontSize=12, leading=17, textColor=MUTED, spaceAfter=16))
    styles.add(ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=TEAL, spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle("BodyCustom", fontName="Helvetica", fontSize=10, leading=14, textColor=INK, spaceAfter=8))
    styles.add(ParagraphStyle("SmallCustom", fontName="Helvetica", fontSize=8, leading=11, textColor=MUTED, spaceAfter=7))
    styles.add(ParagraphStyle("Cell", fontName="Helvetica", fontSize=9, leading=12, textColor=INK))
    styles.add(ParagraphStyle("CellHead", parent=styles["Cell"], fontName="Helvetica-Bold", textColor=colors.white))
    styles.add(ParagraphStyle("CodeCustom", fontName="Helvetica", fontSize=9, leading=13, textColor=INK, backColor=PALE, borderPadding=9, spaceBefore=5, spaceAfter=12))
    story = []

    def p(text, style="BodyCustom"):
        return Paragraph(text, styles[style])

    def body(text):
        story.append(p(text))

    def section(text):
        story.append(p(text, "Section"))

    def source(text):
        story.append(p(escape(text), "SmallCustom"))

    def table(headers, rows, widths):
        data = [[p(escape(str(c)), "CellHead") for c in headers]]
        data += [[p(escape(str(c)), "Cell") for c in row] for row in rows]
        result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, -1), (-1, -1), .5, LINE),
        ]))
        story.append(result)
        story.append(Spacer(1, 6))

    def page(number, title, deck):
        if number > 1:
            story.append(PageBreak())
        source(f"{number:02d}  /  ASSESSMENT REPORT")
        story.append(p(title, "TitleCustom"))
        story.append(p(deck, "Deck"))

    def figure(path, caption):
        image = Image(str(path))
        ratio = image.imageHeight / image.imageWidth
        image.drawWidth, image.drawHeight = WIDTH, WIDTH * ratio
        story.append(image)
        source(caption)

    # Page 1
    page(1, "Freight Rate Prediction", "Machine Learning Engineer Assessment<br/>Affan Amer")
    body("A reproducible solution for estimating a load's posted rate from route, equipment, weight, date, and supplied market signals. The final deliverables contain predictions for every unlabeled load and every fixed December scenario.")
    table(["Reserved main-model MAE", "Baseline MAE", "MAE reduction"], [[f"${final.loc['main', 'mae']:,.2f}", f"${final.loc['reference', 'mae']:,.2f}", f"{improvement:.1f}%"]], [WIDTH / 3] * 3)
    source("Internal out-of-time evaluation: 9,523 September-October loads; models fitted through August. Source: [S3].")
    section("Data and prediction task")
    table(["Dataset", "Rows", "Period / role"], [
        ["Labeled development", "48,000", "January-October 2025; target: posted_rate"],
        ["Final inputs", "12,000", "November-December 2025; no labels supplied"],
        ["Fixed chart inputs", "31", "One scenario for every day of December 2025"],
    ], [145, 60, WIDTH - 205])
    section("Data-quality findings and handling")
    table(["Issue", "Labeled", "Final", "Treatment"], [
        ["Missing weight", "300", "165", "Keep unknown; preserve missingness flag"],
        ["Negative weight", "292", "145", "Set unknown; preserve invalid-value flag"],
        ["Missing market index", "374", "249", "Native numeric missing values plus indicator"],
        ["Unseen city", "-", "1,447 loads", "Unknown-safe model; separate city stress test"],
    ], [110, 54, 70, WIDTH - 234])
    body("IDs are unique and no duplicate feature rows were found. Supplied coordinates are within valid ranges and consistent per city, but were not externally geocoded. Extreme rate labels are retained; a large rate alone is not evidence of an error.")
    source("Source: [S1]. Eight pickup and delivery city names are absent from labeled training. Missing and negative weight counts are disjoint.")

    # Page 2
    page(2, "Validation and features", "Match the future-prediction horizon and keep fitted transformations within training folds.")
    table(["Purpose", "Train", "Validate", "Train / validate rows"], [
        ["Development 1", "Jan-Apr", "May-Jun", "19,110 / 9,696"],
        ["Development 2", "Jan-Jun", "Jul-Aug", "28,806 / 9,671"],
        ["Reserved check", "Jan-Aug", "Sep-Oct", "38,477 / 9,523"],
        ["Final refit", "Jan-Oct", "Nov-Dec (unlabeled)", "48,000 / 12,000"],
    ], [109, 72, 131, WIDTH - 312])
    body("The first two expanding windows provide 19,367 nonoverlapping, out-of-time validation loads for model selection. MAE in dollars is the development selection metric; RMSE and R-squared are also reported because the employer's hidden metric is unspecified. The two-month horizons mirror the final prediction period.")
    body("Model roles and fixed hyperparameters were recorded before inspecting September-October model performance. No early stopping, calibration, feature selection, or model switching used that window. Dataset-level exploration included all labeled months: this is a reserved performance check, not an externally sequestered blind test.")
    section("Leakage and preprocessing controls")
    body("The stateless cleaner does not learn from labels. It excludes load_id and posted_rate, marks nonpositive weights unknown, and adds separate missing/invalid flags. Ridge imputation, scaling and one-hot encoding fit on each training fold only. CatBoost and equipment medians likewise fit only on training rows. No evaluation rows or difficult labels are dropped.")
    section("Two explicit inference paths")
    table(["Path", "Features"], [
        ["Main model", "Pickup/delivery/equipment categories; distance; cleaned weight and flags; supplied coordinates, coordinate deltas and derived coordinate distance; calendar fields; market index and quote signal"],
        ["December chart", "Only pickup, delivery, distance, equipment, weight and date-derived/weight-quality features. Coordinates, market index and quote signal are excluded during both fitting and inference."],
    ], [105, WIDTH - 105])
    body("Calendar inputs include month, weekday, day of month, day of year and weekend. The chart path requires no fabricated future signals or external city lookup. Native categorical handling supports unfamiliar cities; a separate development stress test measures that risk.")
    source("Sources: [S1], [S2], [S4], [S5]. Implementation: freight/data.py, features.py, splits.py, models.py and model_policy.json.")

    # Page 3
    page(3, "Model selection", "Compare simple baselines, objectives and signal availability before the reserved check.")
    names = {
        "median_rate": "Median rate", "equipment_rate_per_mile": "Equipment rate / mile",
        "ridge_core_rate_per_mile": "Ridge core rate / mile", "catboost_core_mae": "CatBoost core - MAE",
        "catboost_market_mae": "CatBoost market - MAE", "catboost_full_mae": "CatBoost full - MAE",
        "catboost_core_rmse": "CatBoost core - RMSE", "catboost_chart_mae": "CatBoost chart - MAE",
    }
    table(["Candidate", "MAE ($)", "RMSE ($)", "R-squared"], [[names[r.model], f"{r.mae:,.2f}", f"{r.rmse:,.2f}", f"{r.r2:.4f}"] for r in leaderboard.itertuples()], [227, 87, 95, WIDTH - 409])
    source("Pooled development windows, 19,367 loads per candidate. Source: [S2]. Core excludes both market index and quote signal; market adds only market index; full adds both.")
    body("The full-feature MAE model has the lowest development MAE ($140.70), improving 34.4% over the equipment baseline. MAE training is less sensitive to extreme labels than squared-error training. The market-only model has slightly lower RMSE, so the tradeoff is retained rather than claiming one candidate dominates every metric.")
    section("Signal stability and geographic robustness")
    body("The quote signal's correlation with actual rate per mile changes sign across months. Its overall correlation hides that change. This motivates explicit signal ablations; it is not proof of leakage. Quote-time availability and signal generation are undocumented.")
    unseen = geographic.loc[geographic.segment.eq("unseen")].sort_values("mae")
    table(["City stress candidate", "Unseen-city MAE ($)"], [[names[r.model], f"{r.mae:,.2f}"] for r in unseen.itertuples()], [315, WIDTH - 315])
    body("Eight cities were chosen by stable, target-independent hashes from the first training window. Removing both route-role occurrences from training produces 4,403 unseen-city evaluation loads across the two windows. The market-only edge is small and comes from one city subset; it does not justify extra city routing. The full model remains the predeclared main model.")
    source("Sources: [S1], [S2], [S4]. All CatBoost candidates: 500 iterations, depth 6, learning rate 0.05, L2 leaf regularization 5, seed 42, four threads; no validation early stopping.")

    # Page 4
    page(4, "Reserved evaluation", "Frozen configurations evaluated on all September-October loads, without another tuning round.")
    labels = {"reference": "Equipment baseline", "main": "Main model", "chart": "Reduced-input chart model"}
    table(["Role", "MAE ($)", "RMSE ($)", "R-squared"], [[labels[role], f"{final.loc[role, 'mae']:,.2f}", f"{final.loc[role, 'rmse']:,.2f}", f"{final.loc[role, 'r2']:.4f}"] for role in ["reference", "main", "chart"]], [227, 87, 95, WIDTH - 409])
    source("N = 9,523. Training N = 38,477. Source: [S3]. The main-model MAE reduction is 49.3% versus the predeclared reference.")
    figure(ROOT / "reports/final_check/final_check.png", "Figure 1. Reserved-check main-model predictions (seeded sample) and monthly MAE. Source: [S3]. The diagonal marks perfect predictions.")
    section("Interpretation and error concentration")
    body("The chart model's slightly lower reserved MAE does not trigger a role switch: the full model was selected on development data and the chart model was selected for reduced-input availability. Both fixed configurations are retained for full-data refitting.")
    full_residual = residuals.loc["catboost_full_mae"]
    body(f"In the earlier development predictions, median absolute error is ${full_residual.median_absolute_error:.2f}, while the largest 1% of errors contribute {100 * full_residual.largest_1pct_squared_error_share:.1f}% of total squared error. This explains why RMSE is much larger than MAE. Every large error remains in the headline scores; tail removal is not used to improve reporting.")
    body("The reserved temporal window contains no unseen cities. The separate city stress results therefore complement, rather than replace, the chronological check. These internal results do not measure accuracy on Spotter's unlabeled final set.")
    source("Sources: [S3], [S4]. Metric arithmetic and source labels were independently checked against the saved row-level predictions.")

    # Page 5
    page(5, "Fixed December predictions", "A separate, validated reduced-input path for the exact scenario supplied with the assessment.")
    table(["Fixed field", "Value"], [
        ["Pickup / delivery", "Lexington / Fort Wayne"],
        ["Distance / equipment / weight", "360 miles / Dry Van / 32,000 lb"],
        ["Dates", "December 1-31, 2025; only date changes"],
    ], [190, WIDTH - 190])
    figure(chart_path, "Figure 2. Exact candidate_december.png generated by the unchanged supplied score.py. The image is embedded without redrawing or modifying the predictions. Source: [S6].")
    december_summary = submission["december_rate_summary"]
    body(f"Across the 31 fixed scenarios, predicted rates range from <b>${december_summary['min']:.2f}</b> to <b>${december_summary['max']:.2f}</b>, with mean <b>${december_summary['mean']:.2f}</b>. These are model outputs, not observed future prices.")
    section("Input availability and inference contract")
    body("The scenario file does not supply pickup/delivery coordinates, market index or quote signal. The chart model excludes those inputs throughout training and temporal evaluation. It uses the same labeled training rows and frozen hyperparameters, with the reduced feature set. The final chart model is fitted on all 48,000 labeled loads.")
    body("The completed file preserves all six original input fields, the seven-column schema, and the original row order. Only predicted_rate is filled. The supplied scorer validates one finite positive prediction for each December date and generates the required chart.")
    section("Scope of the curve")
    body("Date-derived splits can produce weekday and within-month variation, but one partial year does not establish annual seasonality, holiday effects or a calibrated December forecast. No holiday uplift, smoothing or manual date-specific price adjustment is added. The chart path need not equal full-model predictions for similar loads with more inputs.")
    source("Sources: december-chart-inputs.csv, december_predictions.csv, [S3], [S5], [S6].")

    # Page 6
    page(6, "Reproducibility and delivery", "Code, pinned dependencies, native models and independently checked prediction contracts.")
    body("From an activated Python 3.12 environment at the repository root:")
    story.append(Preformatted("python -m pip install -r requirements-lock.txt\npython -m unittest discover -s tests -v\npython -m freight.submission\npython -m freight.audit --require-saved-models", styles["CodeCustom"]))
    body("The generation command fits the unchanged main/chart configurations on all labeled rows, saves native CatBoost models, reloads them, verifies every prediction, fills both CSVs by their contracts, and runs the employer's scorer. Prediction-only mode reuses matching saved models. Input, code, policy, output and model hashes are recorded in manifests.")
    body("A clean source copy and independently installed Windows environment passed all 26 tests, refitted both models from scratch, and reproduced both CSVs and the chart byte for byte. Eleven local integrity checks pass. Hosted Windows/Linux dependency, test, artifact-audit and scorer checks passed; full Linux refit byte identity is not claimed.")
    section("Limitations")
    body("Final labels and the grading metric are unavailable. Signal relationships shift, the geographic test uses one fixed city subset, and heavy rate tails dominate RMSE. No prediction intervals or causal claims are made. Reproductions in another environment retain the frozen configuration and preserve the original evaluation evidence, but exact values across every platform are not guaranteed.")
    section("Artifacts and evidence")
    table(["Reference", "Repository source"], [
        ["S1 - Data audit", "reports/eda/audit.json; reports/eda/exploration.md"],
        ["S2 - Model comparison", "reports/benchmark/leaderboard.csv; fold_metrics.csv"],
        ["S3 - Reserved evaluation", "reports/final_check/metrics.csv; segment_metrics.csv; manifest.json"],
        ["S4 - Robustness", "reports/robustness/location_pooled_metrics.csv; residual_summary.csv"],
        ["S5 - Frozen decisions", "MODEL_DECISIONS.md; model_policy.json"],
        ["S6 - Final artifacts", "validation_predictions.csv; december_predictions.csv; scorer_results/candidate_december.png"],
        ["S7 - Integrity evidence", "reports/audit/checks.json; clean_reproduction.json; reports/submission/manifest.json"],
    ], [128, WIDTH - 128])
    source("Sources are supplied assessment data or generated results in the repository; no external data augments training. All currency values are dollars.")
    body('<b>Repository:</b> <link href="https://github.com/AffanRM/spotter_ml" color="#075563">github.com/AffanRM/spotter_ml</link>')
    source("Submitted prediction files: validation_predictions.csv and december_predictions.csv. Camera-on Loom walkthrough is supplied separately by the candidate.")

    def furniture(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(.6)
        canvas.line(46, A4[1] - 33, A4[0] - 46, A4[1] - 33)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(TEAL)
        canvas.drawString(46, A4[1] - 25, "FREIGHT RATE PREDICTION")
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(46, 26, "Affan Amer  |  Machine Learning Engineer Assessment")
        canvas.drawRightString(A4[0] - 46, 26, str(document.page))
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=46, rightMargin=46,
                                 topMargin=49, bottomMargin=43, title="Freight Rate Prediction - Assessment Report", author="Affan Amer")
    document.build(story, onFirstPage=furniture, onLaterPages=furniture)
    print(f"Created report: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/freight_rate_assessment_report.pdf")
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
