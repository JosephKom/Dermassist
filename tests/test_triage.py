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
    # mel (0.03) + bcc (0.02) = 0.05
    assert pytest.approx(compute_cancer_score(benign_probs), 0.001) == 0.05
    # mel (0.35) + bcc (0.15) = 0.50
    assert pytest.approx(compute_cancer_score(malignant_probs), 0.001) == 0.50


def test_no_escalation_when_all_clear(benign_probs, clean_answers):
    res = floor("Nothing flagged", benign_probs, clean_answers)
    assert res["final_level"] == "Nothing flagged"
    assert not res["escalated"]
    assert len(res["fired_rules"]) == 0


def test_cutoff_escalation(malignant_probs, clean_answers):
    # Gemini says 'Nothing flagged', but cancer score = 0.50 >= 0.20
    res = floor("Nothing flagged", malignant_probs, clean_answers)
    assert res["final_level"] == "Prompt review"
    assert res["escalated"]
    assert any("combined malignant risk" in r for r in res["fired_rules"])


def test_red_flag_symptom_escalation(benign_probs, clean_answers):
    clean_answers["bled"] = True
    res = floor("Routine review", benign_probs, clean_answers)
    assert res["final_level"] == "Prompt review"
    assert res["escalated"]
    assert any("Red-flag symptom reported: bled" in r for r in res["fired_rules"])


def test_never_lowers_level(malignant_probs, clean_answers):
    # If Gemini already deemed 'Prompt review', floor maintains it
    res = floor("Prompt review", malignant_probs, clean_answers)
    assert res["final_level"] == "Prompt review"
    assert not res["escalated"]


def test_cannot_assess_escalation_guard(benign_probs, clean_answers):
    # An unassessed image with active symptoms must still alert the user
    clean_answers["grown"] = True
    res = floor("Cannot assess", benign_probs, clean_answers)
    assert res["final_level"] == "Prompt review"
    assert res["escalated"]


def test_non_escalating_symptom(benign_probs, clean_answers):
    # Itching and pain alone do not hit high-risk escalate thresholds unless configured
    clean_answers["itched"] = True
    clean_answers["hurt"] = True
    res = floor("Nothing flagged", benign_probs, clean_answers)
    assert res["final_level"] == "Nothing flagged"
    assert not res["escalated"]