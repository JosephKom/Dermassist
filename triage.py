import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

# Allowed urgency levels in monotonic ascending order
UrgencyLevel = Literal[
    "Cannot assess",
    "Nothing flagged",
    "Routine review",
    "Prompt review",
]

# Standardized user symptom questionnaire answers
class UserAnswers(TypedDict):
    grown: bool
    changed: bool
    bled: bool
    itched: bool
    hurt: bool

# Path to the shared thresholds configuration in the project root
DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.json"

DEFAULT_CONFIG = {
    "cancer_score_cutoff": 0.20,
    "cancer_classes": ["mel", "bcc"],
    "escalate_on_symptoms": ["grown", "changed", "bled"],
    "urgency_hierarchy": [
        "Cannot assess",
        "Nothing flagged",
        "Routine review",
        "Prompt review",
    ],
}


def load_thresholds(filepath: Optional[Path] = None) -> dict:
    target_path = filepath or DEFAULT_THRESHOLDS_PATH
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG


def compute_cancer_score(
    probabilities: Dict[str, float],
    cancer_classes: Optional[List[str]] = None,
) -> float:
    """Sums the ResNet18 probabilities for specified malignant classes (e.g. mel + bcc)."""
    if cancer_classes is None:
        cancer_classes = ["mel", "bcc"]
    return sum(probabilities.get(cls, 0.0) for cls in cancer_classes)


def floor(
    level: UrgencyLevel,
    probabilities: Dict[str, float],
    answers: UserAnswers,
) -> Tuple[UrgencyLevel, Optional[str]]:
    """
    Applies the rule floor to Gemini's urgency level.
    
    Can escalate the urgency level to 'Prompt review', but never lowers it.
    
    Returns:
        (final_level, fired_rule_message)
        If no floor fired, fired_rule_message is None.
    """
    config = load_thresholds()
    hierarchy: List[str] = config.get("urgency_hierarchy", DEFAULT_CONFIG["urgency_hierarchy"])
    cutoff: float = float(config.get("cancer_score_cutoff", 0.20))
    cancer_classes: List[str] = config.get("cancer_classes", ["mel", "bcc"])
    escalate_symptoms: List[str] = config.get("escalate_on_symptoms", ["grown", "changed", "bled"])

    rank_map = {lvl: idx for idx, lvl in enumerate(hierarchy)}
    base_rank = rank_map.get(level, 0)
    prompt_review_rank = rank_map.get("Prompt review", 3)

    # 1. Compute ResNet18 cancer score (mel + bcc)
    cancer_score = compute_cancer_score(probabilities, cancer_classes)

    fired_rules: List[str] = []

    # 2. Rule: High-risk red-flag symptoms
    active_red_flags = [s for s in escalate_symptoms if answers.get(s, False)]
    if active_red_flags:
        fired_rules.append(f"red-flag symptom reported ({', '.join(active_red_flags)})")

    # 3. Rule: ResNet18 combined cancer score reaches or exceeds cutoff
    if cancer_score >= cutoff:
        fired_rules.append(
            f"ResNet18 malignant risk score ({cancer_score:.2%}) >= threshold ({cutoff:.2%})"
        )

    # 4. Check if rules escalate the level
    if fired_rules and base_rank < prompt_review_rank:
        reason_msg = "Urgency raised to Prompt review: " + "; ".join(fired_rules)
        return "Prompt review", reason_msg

    # No escalation triggered, or Gemini is already at or above target
    return level, None