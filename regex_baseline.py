"""
Regex-based baseline for pregnancy-stage classification from obstetric
ultrasound reports.

Three-class classification:
  - "early active pregnancy"                (GA < 14 weeks)
  - "late active pregnancy"                 (GA >= 14 weeks)
  - "no active pregnancy"

Design rationale
----------------
This baseline is deliberately strong: it encodes the same clinical decision
rules a domain expert would use, translated into
pattern matching. The goal is to establish an upper bound on what rule-based
approaches can achieve on this task, so that any advantage of LLM-based
methods is clearly attributable to language understanding rather than simple
keyword presence.

The classifier applies rules in the following priority order (see the
`# --- Rule N ---` comments in `classify_report_detailed`):
  1. Unrelated indicators    (negation of pregnancy) - fires ONLY when there
                             is no gestational age and no early or late
                             markers, i.e. when `is_unrelated` is true and
                             `early_count == 0` and `late_count == 0` and
                             `ga_weeks is None`
  2. Explicit GA extraction  (weeks -> early if GA < 14, else late)
  3. Trimester keywords      (direct mentions)
  4. CRL measurement         (crown-rump length implies early)
  5. Marker-count comparison (early-pregnancy markers such as yolk sac and
                             gestational sac vs late-pregnancy markers such
                             as biometry, BPP, Dopplers and presentation)
  6. EDD/EDC presence        (implies a known ongoing pregnancy -> late)
  7. Unrelated indicators    (with weak pregnancy evidence)
  8. Fallback                (no signal -> no active pregnancy)

Note that the unrelated check is consulted twice: once at step 1 as a
high-confidence negation that requires the absence of any pregnancy signal,
and again at step 7 as a lower-confidence fallback once the positive rules
have all declined to fire.

OCR robustness: patterns account for common OCR errors (spaces in numbers,
missing punctuation, inconsistent abbreviations, digit/letter confusion).

Usage
-----
    from regex_baseline import classify_report, evaluate_corpus

    # Single report
    label = classify_report(report_text)

    # Batch evaluation against gold labels
    results = evaluate_corpus(texts, gold_labels)
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
EARLY = "early active pregnancy"
LATE = "late active pregnancy"
UNRELATED = "no active pregnancy"

EARLY_CUTOFF_WEEKS = 14  # < 14 weeks = early; >= 14 weeks = late


# ---------------------------------------------------------------------------
# 1. Gestational age extraction
# ---------------------------------------------------------------------------

# Matches patterns like:
#   "12 weeks 3 days", "12 wks 3 days", "12w3d", "12 W 3 D",
#   "12 weeks", "GA 34 weeks", "gestational age 12 weeks",
#   "D- 12 Wks 6 Days", "GA: 28w2d", "28 wk 3 d",
#   "age = 12 weeks", "age - 12 wks"
#   OCR variants: extra spaces, missing separators

_GA_WEEKS_DAYS = re.compile(
    r"""
    (?:                                     # optional GA prefix
        (?:gestational\s*age|ga|dating|age|d)
        \s*[-:=~]?\s*
    )?
    (\d{1,2})                               # weeks (group 1)
    \s*(?:weeks?|wks?|w)\s*                 # week unit
    (?:                                     # optional days part
        [,\s]*
        (\d{1})                             # days (group 2)
        \s*(?:days?|d)\b
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matches "GA 12+3", "12+5 weeks" (weeks+days with plus notation)
_GA_PLUS_NOTATION = re.compile(
    r"""
    (?:(?:gestational\s*age|ga|dating|age|d)\s*[-:=~]?\s*)?
    (\d{1,2})                               # weeks (group 1)
    \s*\+\s*
    (\d{1})                                 # days (group 2)
    (?:\s*(?:weeks?|wks?|w))?               # optional unit
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matches "XX weeks gestation/pregnancy/gravid"
_WEEKS_GESTATION = re.compile(
    r"""
    (\d{1,2})\s*(?:weeks?|wks?)\s+
    (?:gestation|pregnancy|gravid|pregnant|gest)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matches trimester-based GA like "first trimester" → early,
# "second trimester" or "third trimester" → late
_TRIMESTER = re.compile(
    r"""
    (first|1st|second|2nd|third|3rd)
    \s+trimester
    """,
    re.IGNORECASE | re.VERBOSE,
)

# EDD/EDC patterns that imply pregnancy but don't directly give GA
_EDD_PATTERN = re.compile(
    r"\b(?:edd|edc|expected\s+(?:date|delivery))\b",
    re.IGNORECASE,
)


def _extract_ga_weeks(text: str) -> Optional[float]:
    """
    Extract gestational age in weeks from text.
    Returns float (e.g., 12.857 for 12w6d) or None if not found.
    Tries multiple patterns and returns the *first* match found
    (priority: explicit GA label > weeks+days > plus notation > weeks gestation).
    """
    # Try patterns with explicit GA prefix first (higher confidence)
    for pattern in [_GA_WEEKS_DAYS, _GA_PLUS_NOTATION, _WEEKS_GESTATION]:
        for match in pattern.finditer(text):
            weeks_str = match.group(1)
            try:
                weeks = int(weeks_str)
            except (ValueError, TypeError):
                continue

            # Sanity check: gestational age should be 1-45 weeks
            if weeks < 1 or weeks > 45:
                continue

            days = 0
            if pattern != _WEEKS_GESTATION and match.lastindex >= 2:
                days_str = match.group(2)
                if days_str is not None:
                    try:
                        days = int(days_str)
                    except (ValueError, TypeError):
                        days = 0

            return weeks + days / 7.0

    return None


def _extract_trimester(text: str) -> Optional[str]:
    """Extract explicit trimester mention."""
    match = _TRIMESTER.search(text)
    if match:
        tri = match.group(1).lower()
        if tri in ("first", "1st"):
            return EARLY
        elif tri in ("second", "2nd", "third", "3rd"):
            return LATE
    return None


# ---------------------------------------------------------------------------
# 2. Unrelated / no-pregnancy indicators
# ---------------------------------------------------------------------------

_UNRELATED_PATTERNS = [
    # Explicit negation of pregnancy
    re.compile(r"\b(?:not\s+pregnant|non[\s-]*pregnant|no\s+pregnancy)\b", re.I),
    # Postpartum / post-delivery
    re.compile(r"\b(?:postpartum|post[\s-]*partum|post[\s-]*delivery|post[\s-]*natal"
               r"|postnatal|post[\s-]*cesarean|post[\s-]*c[\s-]*section)\b", re.I),
    # Gynecological (non-obstetric) focus
    re.compile(r"\b(?:gynecol|gynaecol|gyn\s+exam|pelvic\s+(?:exam|scan|ultrasound)"
               r"|transvaginal\s+(?:scan|exam)|follicle\s+(?:study|monitoring)"
               r"|follicular\s+study|ovarian\s+(?:cyst|mass|torsion)"
               r"|fibroid|leiomyoma|endometri(?:osis|al\s+thickness)"
               r"|iud|intrauterine\s+device|copper\s+t)\b", re.I),
    # Negative pregnancy test
    re.compile(r"\b(?:pregnancy\s+test\s+negative|beta[\s-]*hcg\s+negative"
               r"|negative\s+pregnancy|hcg\s*[-:]?\s*negative)\b", re.I),
    # Infertility workup
    re.compile(r"\b(?:infertility|ivf\s+(?:workup|evaluation)|iui\s+monitoring)\b", re.I), 
    # Empty uterus / no gestational sac
    re.compile(r"\b(?:empty\s+uterus|no\s+gestational\s+sac|no\s+intrauterine"
               r"|no\s+evidence\s+of\s+(?:pregnancy|gestational))\b", re.I),
]


def _is_unrelated(text: str) -> bool:
    """Check if text indicates no current viable pregnancy."""
    for pattern in _UNRELATED_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# 3. Early-pregnancy markers (< 14 weeks typical findings)
# ---------------------------------------------------------------------------

_EARLY_MARKERS = [
    # CRL (crown-rump length) — the hallmark early measurement
    re.compile(r"\b(?:crl|crown[\s-]*rump(?:\s+length)?)\b", re.I),
    # Yolk sac
    re.compile(r"\byolk\s*sac\b", re.I),
    # Gestational sac (without biometry suggesting it's an early scan)
    re.compile(r"\bgestational\s*sac\b", re.I),
    # Fetal pole
    re.compile(r"\bfetal\s*pole\b", re.I),
    # NT (nuchal translucency) — first trimester screening
    re.compile(r"\b(?:nuchal\s*translucency|nt\s*[-:=]\s*\d)\b", re.I),
    # Nasal bone assessment (first trimester screening)
    re.compile(r"\bnasal\s+bone\s+(?:seen|present|absent|not\s+seen)\b", re.I),
    # Ductus venosus (first trimester Doppler)
    re.compile(r"\bductus\s*venosus\b", re.I),
    # Double decidual sign
    re.compile(r"\bdouble\s+decidual\b", re.I),
    # Viability scan
    re.compile(r"\bviability\s*(?:scan|check|assessment)\b", re.I),
    # Dating scan
    re.compile(r"\bdating\s*(?:scan|ultrasound)\b", re.I),
]


def _count_early_markers(text: str) -> int:
    """Count early-pregnancy marker matches."""
    return sum(1 for p in _EARLY_MARKERS if p.search(text))


# ---------------------------------------------------------------------------
# 4. Late-pregnancy markers (>= 14 weeks typical findings)
# ---------------------------------------------------------------------------

_LATE_MARKERS = [
    # Standard biometry (BPD, HC, AC, FL)
    re.compile(r"\b(?:bpd|biparietal)\b", re.I),
    re.compile(r"\b(?:hc|head\s*circumference)\b", re.I),
    re.compile(r"\b(?:ac|abdominal\s*circumference)\b", re.I),
    re.compile(r"\b(?:fl|femur\s*length)\b", re.I),
    # Estimated fetal weight
    re.compile(r"\b(?:efw|estimated\s+fetal\s+weight|fetal\s+weight)\b", re.I),
    # Biophysical profile
    re.compile(r"\b(?:bpp|biophysical\s+profile)\b", re.I),
    # Fetal presentation / lie
    re.compile(r"\b(?:cephalic|breech|transverse\s+lie|vertex|presentation"
               r"|longitudinal\s+lie)\b", re.I),
    # Placenta details (location, grade, previa)
    re.compile(r"\b(?:placenta\s+(?:anterior|posterior|fundal|fundo|lateral|low[\s-]*lying"
               r"|previa|praevia|grade)|placental\s+(?:location|grading|maturity))\b", re.I),
    # Amniotic fluid index / deepest vertical pocket
    re.compile(r"\b(?:afi|amniotic\s+fluid\s+index|deepest\s+(?:vertical\s+)?pocket"
               r"|dvp|liquor\s+(?:adequate|reduced|increased|normal))\b", re.I),
    # Cervical length in late pregnancy context
    re.compile(r"\bcervical\s+length\b", re.I),
    # Doppler studies (umbilical, MCA, uterine artery in late context)
    re.compile(r"\b(?:umbilical\s+artery|ua\s+(?:pi|ri|s/d)|mca\s+(?:pi|ri|psv)"
               r"|middle\s+cerebral\s+artery|cerebroplacental\s+ratio|cpr)\b", re.I),
    # Fetal anatomy survey
    re.compile(r"\b(?:anatomy\s+(?:scan|survey|screening)|anomaly\s+scan"
               r"|structural\s+(?:scan|survey)|morphology\s+scan)\b", re.I),
    # Growth scan / growth assessment
    re.compile(r"\b(?:growth\s+(?:scan|assessment|surveillance|monitoring)"
               r"|interval\s+growth)\b", re.I),
    # Fetal movements
    re.compile(r"\bfetal\s+movements?\b", re.I),
    # Cord details
    re.compile(r"\b(?:cord\s+(?:insertion|vessels|around\s+neck|nuchal)"
               r"|(?:two|three|2|3)\s+vessel\s+cord)\b", re.I),
]


def _count_late_markers(text: str) -> int:
    """Count late-pregnancy marker matches."""
    return sum(1 for p in _LATE_MARKERS if p.search(text))


# ---------------------------------------------------------------------------
# 5. CRL-based GA inference (fallback)
# ---------------------------------------------------------------------------

# CRL measurement in mm — if present without explicit GA, infer stage.
# CRL is used up to ~84mm (~14 weeks). Typical: CRL 45mm ≈ 11 weeks.
_CRL_VALUE = re.compile(
    r"\b(?:crl|crown[\s-]*rump)\s*[-:=]?\s*(\d{1,3})\s*(?:mm|cm)?\b",
    re.IGNORECASE,
)


def _extract_crl_mm(text: str) -> Optional[float]:
    """Extract CRL measurement in mm."""
    match = _CRL_VALUE.search(text)
    if match:
        val = float(match.group(1))
        # If value is small (likely cm), convert to mm
        if val < 15:
            val *= 10
        # Sanity: CRL should be roughly 1-120mm
        if 1 <= val <= 120:
            return val
    return None


# ---------------------------------------------------------------------------
# 6. Main classifier
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Structured output from the rule-based classifier."""
    label: str
    confidence: str  # "high", "medium", "low"
    rule_fired: str  # which rule determined the label
    ga_weeks: Optional[float] = None
    early_marker_count: int = 0
    late_marker_count: int = 0
    details: dict = field(default_factory=dict)


def classify_report(text: str) -> str:
    """
    Classify a single obstetric ultrasound report.

    Parameters
    ----------
    text : str
        Raw OCR-derived report text.

    Returns
    -------
    str
        One of: "early pregnancy", "late pregnancy",
        "unrelated or no current pregnancy"
    """
    return classify_report_detailed(text).label


def classify_report_detailed(text: str) -> ClassificationResult:
    """
    Classify with full diagnostic output.

    Priority order:
      1. Unrelated indicators → unrelated, only when there is no GA and no
         early or late markers (is_unrelated and early_count == 0 and
         late_count == 0 and ga_weeks is None)
      2. Explicit GA in weeks → early/late by cutoff
      3. Explicit trimester mention
      4. CRL measurement → infer early
      5. Marker count comparison (early vs late markers)
      6. EDD/EDC presence → late (implies known ongoing pregnancy)
      7. Unrelated flag with weak pregnancy evidence → unrelated
      8. Fallback: no signal → unrelated
    """
    if not text or not text.strip():
        return ClassificationResult(
            label=UNRELATED,
            confidence="high",
            rule_fired="empty_input",
        )

    # Normalize whitespace for OCR robustness
    clean = re.sub(r"\s+", " ", text).strip()

    early_count = _count_early_markers(clean)
    late_count = _count_late_markers(clean)
    ga_weeks = _extract_ga_weeks(clean)
    crl_mm = _extract_crl_mm(clean)
    trimester = _extract_trimester(clean)
    has_edd = bool(_EDD_PATTERN.search(clean))
    is_unrelated = _is_unrelated(clean)

    details = {
        "ga_weeks": ga_weeks,
        "crl_mm": crl_mm,
        "trimester_mention": trimester,
        "has_edd": has_edd,
        "early_markers": early_count,
        "late_markers": late_count,
        "unrelated_flag": is_unrelated,
    }

    # --- Rule 1: Unrelated indicators ---
    # Only fire if there are NO strong pregnancy indicators
    if is_unrelated and early_count == 0 and late_count == 0 and ga_weeks is None:
        return ClassificationResult(
            label=UNRELATED,
            confidence="high",
            rule_fired="unrelated_no_pregnancy_indicators",
            early_marker_count=early_count,
            late_marker_count=late_count,
            details=details,
        )

    # If unrelated flag fires BUT we also have pregnancy indicators,
    # the unrelated flag might be incidental (e.g., "postpartum" in
    # history section but current scan shows ongoing pregnancy).
    # We continue to other rules but note the conflict.

    # --- Rule 2: Explicit GA extraction ---
    if ga_weeks is not None:
        if ga_weeks < EARLY_CUTOFF_WEEKS:
            return ClassificationResult(
                label=EARLY,
                confidence="high",
                rule_fired=f"explicit_ga_{ga_weeks:.1f}w",
                ga_weeks=ga_weeks,
                early_marker_count=early_count,
                late_marker_count=late_count,
                details=details,
            )
        else:
            return ClassificationResult(
                label=LATE,
                confidence="high",
                rule_fired=f"explicit_ga_{ga_weeks:.1f}w",
                ga_weeks=ga_weeks,
                early_marker_count=early_count,
                late_marker_count=late_count,
                details=details,
            )

    # --- Rule 3: Explicit trimester ---
    if trimester is not None:
        return ClassificationResult(
            label=trimester,
            confidence="medium",
            rule_fired="trimester_keyword",
            early_marker_count=early_count,
            late_marker_count=late_count,
            details=details,
        )

    # --- Rule 4: CRL measurement (implies early) ---
    if crl_mm is not None:
        # CRL is a first-trimester measurement
        return ClassificationResult(
            label=EARLY,
            confidence="medium",
            rule_fired=f"crl_measurement_{crl_mm:.0f}mm",
            early_marker_count=early_count,
            late_marker_count=late_count,
            details=details,
        )

    # --- Rule 5: Marker count comparison ---
    total_markers = early_count + late_count

    if total_markers > 0:
        # Strong late signal
        if late_count >= 3 and late_count > early_count:
            return ClassificationResult(
                label=LATE,
                confidence="medium",
                rule_fired=f"late_markers_{late_count}_vs_early_{early_count}",
                early_marker_count=early_count,
                late_marker_count=late_count,
                details=details,
            )

        # Strong early signal
        if early_count >= 2 and early_count > late_count:
            return ClassificationResult(
                label=EARLY,
                confidence="medium",
                rule_fired=f"early_markers_{early_count}_vs_late_{late_count}",
                early_marker_count=early_count,
                late_marker_count=late_count,
                details=details,
            )

        # Mixed signals — use ratio with slight late bias
        # (late markers are more numerous in the pattern set, so
        #  even a small count is meaningful)
        if late_count >= 2:
            return ClassificationResult(
                label=LATE,
                confidence="low",
                rule_fired=f"late_markers_weak_{late_count}",
                early_marker_count=early_count,
                late_marker_count=late_count,
                details=details,
            )

        if early_count >= 1:
            return ClassificationResult(
                label=EARLY,
                confidence="low",
                rule_fired=f"early_markers_weak_{early_count}",
                early_marker_count=early_count,
                late_marker_count=late_count,
                details=details,
            )

    # --- Rule 6: EDD implies ongoing pregnancy, likely late ---
    if has_edd:
        return ClassificationResult(
            label=LATE,
            confidence="low",
            rule_fired="edd_present",
            early_marker_count=early_count,
            late_marker_count=late_count,
            details=details,
        )

    # --- Rule 7: Unrelated flag with weak pregnancy evidence ---
    if is_unrelated:
        return ClassificationResult(
            label=UNRELATED,
            confidence="medium",
            rule_fired="unrelated_weak_evidence",
            early_marker_count=early_count,
            late_marker_count=late_count,
            details=details,
        )

    # --- Fallback: no signal → unrelated ---
    return ClassificationResult(
        label=UNRELATED,
        confidence="low",
        rule_fired="no_signal_fallback",
        early_marker_count=early_count,
        late_marker_count=late_count,
        details=details,
    )


# ---------------------------------------------------------------------------
# 7. Evaluation utilities
# ---------------------------------------------------------------------------

def evaluate_corpus(
    texts: list[str],
    gold_labels: list[str],
    verbose: bool = False,
) -> dict:
    """
    Evaluate the regex baseline on a labeled corpus.

    Parameters
    ----------
    texts : list of str
        Report texts.
    gold_labels : list of str
        Ground truth labels (must match label constants).
    verbose : bool
        If True, print per-document diagnostics for errors.

    Returns
    -------
    dict with keys:
        "accuracy", "macro_f1", "per_class" (precision/recall/f1),
        "confusion_matrix", "predictions", "detailed_results",
        "error_analysis"
    """
    from collections import Counter

    assert len(texts) == len(gold_labels), "texts and labels must have same length"

    labels = [EARLY, LATE, UNRELATED]
    label_set = set(labels)

    predictions = []
    detailed = []
    errors = []

    for i, (text, gold) in enumerate(zip(texts, gold_labels)):
        result = classify_report_detailed(text)
        predictions.append(result.label)
        detailed.append(result)

        if result.label != gold:
            errors.append({
                "index": i,
                "gold": gold,
                "predicted": result.label,
                "rule_fired": result.rule_fired,
                "confidence": result.confidence,
                "details": result.details,
                "text_snippet": text[:200],
            })

    # Confusion matrix
    confusion = {true: {pred: 0 for pred in labels} for true in labels}
    for gold, pred in zip(gold_labels, predictions):
        if gold in label_set and pred in label_set:
            confusion[gold][pred] += 1

    # Per-class metrics
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    # Macro F1
    macro_f1 = sum(per_class[l]["f1"] for l in labels) / len(labels)

    # Accuracy
    correct = sum(1 for g, p in zip(gold_labels, predictions) if g == p)
    accuracy = correct / len(gold_labels) if gold_labels else 0.0

    # Error analysis summary
    error_rules = Counter(e["rule_fired"] for e in errors)
    error_types = Counter((e["gold"], e["predicted"]) for e in errors)

    if verbose:
        print(f"\n{'='*60}")
        print(f"REGEX BASELINE EVALUATION")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f} ({correct}/{len(gold_labels)})")
        print(f"Macro F1:  {macro_f1:.4f}")
        print(f"\nPer-class metrics:")
        for label in labels:
            m = per_class[label]
            print(f"  {label:45s}  P={m['precision']:.3f}  "
                  f"R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                  f"n={m['support']}")
        print(f"\nConfusion matrix (rows=gold, cols=predicted):")
        header = f"{'':45s} {'early':>8s} {'late':>8s} {'unrelated':>10s}"
        print(header)
        for true_label in labels:
            row = f"{true_label:45s}"
            for pred_label in labels:
                row += f" {confusion[true_label][pred_label]:>8d}"
            print(row)
        print(f"\nTotal errors: {len(errors)}")
        if error_types:
            print(f"Most common error types (gold → pred):")
            for (g, p), count in error_types.most_common(5):
                print(f"  {g} → {p}: {count}")
        if error_rules:
            print(f"Rules that produced errors:")
            for rule, count in error_rules.most_common(10):
                print(f"  {rule}: {count}")

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "predictions": predictions,
        "detailed_results": detailed,
        "error_analysis": {
            "total_errors": len(errors),
            "errors": errors,
            "error_rules": dict(error_rules),
            "error_types": {f"{g} -> {p}": c for (g, p), c in error_types.items()},
        },
    }


# ---------------------------------------------------------------------------
# 8. Quick self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Smoke test with representative examples."""
    test_cases = [
        # Early pregnancy: explicit GA
        (
            "Single live fetus is seen. CRL- 64mm D- 12 Wks 6 Days "
            "NT- 1.3mm Nasal bone is seen. FHR- 157/bpm.",
            EARLY,
        ),
        # Early pregnancy: CRL without explicit GA
        (
            "Single gestational sac with yolk sac seen. CRL 8mm. "
            "Fetal cardiac activity present.",
            EARLY,
        ),
        # Late pregnancy: explicit GA
        (
            "Single live fetus in cephalic presentation. GA 32 weeks 4 days. "
            "BPD 81mm HC 295mm AC 278mm FL 62mm. EFW 1850g. "
            "Placenta posterior. AFI 12cm.",
            LATE,
        ),
        # Late pregnancy: biometry without explicit GA
        (
            "BPD 76mm. HC 280mm. AC 260mm. FL 58mm. Cephalic presentation. "
            "Placenta anterior grade II. Amniotic fluid index adequate.",
            LATE,
        ),
        # Unrelated: gynecological
        (
            "Transvaginal scan performed. Uterus anteverted, normal size. "
            "Endometrial thickness 6mm. Right ovarian cyst 3cm. "
            "No free fluid in POD.",
            UNRELATED,
        ),
        # Unrelated: not pregnant
        (
            "Patient not pregnant. Pelvic ultrasound for fibroid assessment. "
            "Multiple fibroids noted.",
            UNRELATED,
        ),
        # Late pregnancy: GA in plus notation
        (
            "Fetal biometry consistent with GA 28+3. Presentation cephalic. "
            "Umbilical artery PI normal.",
            LATE,
        ),
        # Early: first trimester keywords
        (
            "First trimester screening scan. Single viable intrauterine pregnancy.",
            EARLY,
        ),
        # Unrelated: missed abortion
        (
            "No fetal cardiac activity detected. Missed abortion. "
            "Products of conception seen.",
            UNRELATED,
        ),
        # Late: weeks gestation pattern
        (
            "Patient at 36 weeks gestation. Fetal movements present. "
            "Breech presentation noted.",
            LATE,
        ),
    ]

    print("Running self-test...")
    passed = 0
    for i, (text, expected) in enumerate(test_cases):
        result = classify_report_detailed(text)
        status = "PASS" if result.label == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            print(f"  [{status}] Case {i}: expected '{expected}', "
                  f"got '{result.label}' (rule: {result.rule_fired})")
            print(f"         Text: {text[:100]}...")

    print(f"Self-test: {passed}/{len(test_cases)} passed\n")
    return passed == len(test_cases)


if __name__ == "__main__":
    _self_test()
