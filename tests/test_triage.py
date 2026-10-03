import pytest

from triage import compute_cancer_score, floor


@pytest.fixture
def benign_probs():
    return {
        "akiec": 0.01,
        "bcc": 0.02,
        "bkl": 0.05,
        "df": 0.01,
        "mel": 0.03,
        "nv": 0.87,
        "vasc": 0.01,
    }


@pytest.fixture
def malignant_probs():
    return {
        "akiec": 0.05,
        "bcc": 0.15,
        "bkl": 0.05,
        "df": 0.02,
        "mel": 0.35,
        "nv": 0.37,
        "vasc": 0.01,
    }


@pytest.fixture
def clean_answers():
    return {
        "grown": False,
        "changed": False,
        "bled": False,
        "itched": False,
        "hurt": False,
    }


def test_cancer_score_calculation(benign_probs, malignant_probs):
    assert compute_cancer_score(benign_probs) == pytest.approx(0.05)
    assert compute_cancer_score(malignant_probs) == pytest.approx(0.50)


def test_no_escalation_when_all_clear(benign_probs, clean_answers):
    assert floor("Nothing flagged", benign_probs, clean_answers) == (
        "Nothing flagged",
        None,
    )


def test_cutoff_escalation(malignant_probs, clean_answers):
    level, reason = floor("Nothing flagged", malignant_probs, clean_answers)
    assert level == "Prompt review"
    assert reason is not None
    assert "malignant risk score" in reason


def test_red_flag_symptom_escalation(benign_probs, clean_answers):
    clean_answers["bled"] = True
    level, reason = floor("Routine review", benign_probs, clean_answers)
    assert level == "Prompt review"
    assert reason is not None
    assert "bled" in reason


def test_never_lowers_level(malignant_probs, clean_answers):
    assert floor("Prompt review", malignant_probs, clean_answers) == (
        "Prompt review",
        None,
    )


def test_cannot_assess_escalates_on_red_flag(benign_probs, clean_answers):
    clean_answers["grown"] = True
    level, reason = floor("Cannot assess", benign_probs, clean_answers)
    assert level == "Prompt review"
    assert reason is not None


def test_non_escalating_symptom(benign_probs, clean_answers):
    clean_answers["itched"] = True
    clean_answers["hurt"] = True
    assert floor("Nothing flagged", benign_probs, clean_answers) == (
        "Nothing flagged",
        None,
    )


def test_none_level_starts_at_nothing_flagged(benign_probs, clean_answers):
    assert floor(None, benign_probs, clean_answers) == ("Nothing flagged", None)
