"""DermaAssist web app. Start with: python app.py

Pipeline: predict (ResNet18) -> assess (Gemini) -> floor (urgency rules) -> render.
Nothing is saved to disk and inputs are never logged.
"""

import html
from pathlib import Path

import gradio as gr

# Agreed urgency levels (see CONTRACTS.md; must match triage.py and gemini_assess.py).
CANNOT_ASSESS = "Cannot assess"
PROMPT_REVIEW = "Prompt review"
ROUTINE_REVIEW = "Routine review"
NOTHING_FLAGGED = "Nothing flagged"

CLASS_NAMES = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus (mole)",
    "vasc": "Vascular lesion",
}

QUESTIONS = {
    "grown": "Has it grown?",
    "changed": "Has it changed shape or colour?",
    "bled": "Has it bled?",
    "itched": "Does it itch?",
    "hurt": "Does it hurt?",
}

PROTOTYPE_NOTICE = (
    "**Research prototype. Not a medical device and not a diagnosis.** "
    "Always see a doctor about a spot that worries you."
)
UPLOAD_NOTICE = "Your image is sent to Google Gemini for analysis. Nothing is saved."

EXAMPLES_DIR = Path(__file__).parent / "examples"


# --- Teammates' modules (signatures in CONTRACTS.md), with temporary stand-ins until merged. ---
# TODO: delete the stand-ins once predictor.py, gemini_assess.py and triage.py are on main.

try:
    from predictor import predict
except ImportError:
    def predict(image):
        return {"nv": 0.55, "mel": 0.20, "bkl": 0.12, "bcc": 0.06,
                "akiec": 0.04, "df": 0.02, "vasc": 0.01}

try:
    from gemini_assess import assess
except ImportError:
    def assess(image, probabilities, answers):
        return {
            "is_skin_lesion": True,
            "image_issue": None,
            "diagnosis": "Melanocytic nevus",
            "confidence": "medium",
            "visible_features": "Brown, fairly even colour; regular border; roughly symmetric.",
            "urgency_level": ROUTINE_REVIEW,
            "urgency_reasons": ["Stand-in reply: Gemini step not merged yet."],
            "treatment_info": "Stand-in reply: usually monitored; removed if it changes.",
        }

try:
    from triage import floor
except ImportError:
    def floor(level, probabilities, answers):
        level = level or NOTHING_FLAGGED
        if level == PROMPT_REVIEW:
            return level, None
        if answers.get("bled"):
            return PROMPT_REVIEW, "A red-flag answer (bleeding) was given."
        return level, None


# --- Pipeline ---

def run_assessment(image, grown, changed, bled, itched, hurt):
    if image is None:
        return (render_message("Please upload an image."), gr.update(visible=False),
                "", None, "")

    answers = {"grown": grown, "changed": changed, "bled": bled,
               "itched": itched, "hurt": hurt}
    probs = predict(image)

    try:
        gemini = assess(image, probs, answers)
    except Exception:
        gemini = None

    if gemini and not gemini.get("is_skin_lesion", True):
        # Scope check failed: report it and skip the rule floor.
        # TODO: confirm with Antonio whether red-flag answers should still escalate here.
        final_level, rule = CANNOT_ASSESS, None
    else:
        level = gemini.get("urgency_level") if gemini else None
        final_level, rule = floor(level, probs, answers)

    show_gemini = gemini is not None and final_level != CANNOT_ASSESS
    return (
        render_banner(final_level, gemini, rule),
        gr.update(visible=show_gemini),
        render_gemini(gemini, probs) if show_gemini else "",
        # Hidden for Cannot assess: ResNet18 names a lesion even for a photo of a wall.
        None if final_level == CANNOT_ASSESS
        else {CLASS_NAMES.get(k, k): v for k, v in probs.items()},
        render_treatment(gemini) if show_gemini else "",
    )


# --- Rendering ---

# Fixed message and colour per level. "Nothing flagged" is deliberately not green.
LEVEL_STYLE = {
    PROMPT_REVIEW: ("#d93025", "See a doctor soon."),
    ROUTINE_REVIEW: ("#e8a200", "Book a skin check."),
    NOTHING_FLAGGED: ("#5f6b7a", "Nothing was flagged. This is not an all-clear. "
                                 "See a doctor if the spot changes or worries you."),
    CANNOT_ASSESS: ("#8a8f98", "The tool cannot judge this image. "
                               "Retake it, or see a doctor if concerned."),
}

# Keywords that map Gemini's free-text diagnosis onto the seven ResNet18 classes.
# Order matters: more specific terms come first.
DIAGNOSIS_KEYWORDS = [
    ("mel", ["melanoma"]),
    ("bcc", ["basal cell"]),
    ("akiec", ["actinic", "solar keratosis", "bowen", "intraepithelial", "squamous"]),
    ("bkl", ["seborrheic", "seborrhoeic", "benign keratosis", "lentigo", "lichen planus"]),
    ("df", ["dermatofibroma"]),
    ("vasc", ["vascular", "angioma", "haemangioma", "hemangioma", "pyogenic"]),
    ("nv", ["nevus", "naevus", "nevi", "naevi", "mole"]),
]


def diagnosis_to_class(diagnosis):
    text = (diagnosis or "").lower()
    for code, keywords in DIAGNOSIS_KEYWORDS:
        if any(k in text for k in keywords):
            return code
    return None


def banner_html(colour, inner):
    # A tinted background keeps the text readable in both light and dark themes.
    style = (f"border-left:6px solid {colour};"
             f"background:color-mix(in srgb, {colour} 14%, transparent);"
             "border-radius:8px;padding:12px 16px;")
    return f'<div style="{style}">{inner}</div>'


def render_message(text):
    return banner_html(LEVEL_STYLE[CANNOT_ASSESS][0], f"<p>{html.escape(text)}</p>")


def render_banner(level, gemini, rule):
    colour, message = LEVEL_STYLE.get(level, LEVEL_STYLE[CANNOT_ASSESS])
    parts = [f"<h2>{html.escape(level)}</h2>", f"<p><strong>{html.escape(message)}</strong></p>"]

    if gemini is None:
        parts.append("<p><em>AI assessment unavailable. "
                     "Showing the image model result only.</em></p>")
    elif level == CANNOT_ASSESS and gemini.get("image_issue"):
        parts.append(f"<p>{html.escape(gemini['image_issue'])}</p>")

    reasons = []
    if gemini is not None:
        raw_reasons = gemini.get("urgency_reasons") or []
        reasons = raw_reasons if isinstance(raw_reasons, list) else [raw_reasons]
    if rule:
        reasons.append(f"Raised by safety rule: {rule}")
    if reasons:
        parts.append("<ul>" + "".join(f"<li>{html.escape(str(r))}</li>" for r in reasons) + "</ul>")

    return banner_html(colour, "".join(parts))


def render_gemini(gemini, probs):
    lines = [
        f"**Diagnosis:** {gemini.get('diagnosis') or 'Not given'}",
        f"**Confidence:** {gemini.get('confidence') or 'Not given'}",
        f"**Visible features:** {gemini.get('visible_features') or 'Not given'}",
    ]
    gemini_class = diagnosis_to_class(gemini.get("diagnosis"))
    resnet_top = max(probs, key=probs.get)
    if gemini_class is None:
        lines.append("ℹ️ Gemini's diagnosis is not one of the seven conditions "
                     "the image model knows, so they cannot be compared directly.")
    elif gemini_class != resnet_top:
        lines.append(f"⚠️ **The two models disagree.** The image model's top answer is "
                     f"{CLASS_NAMES[resnet_top]}.")
    return "\n\n".join(lines)


def render_treatment(gemini):
    if not gemini.get("treatment_info"):
        return ""
    return "### How this is usually treated (general information)\n\n" + gemini["treatment_info"]


# --- UI ---

with gr.Blocks(title="DermaAssist") as demo:
    gr.Markdown("# DermaAssist")
    gr.Markdown(PROTOTYPE_NOTICE)

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", sources=["upload"], label="Close-up of the skin spot")
            gr.Markdown(UPLOAD_NOTICE)
            checkboxes = [gr.Checkbox(label=q) for q in QUESTIONS.values()]
            submit = gr.Button("Assess", variant="primary")
            if EXAMPLES_DIR.is_dir():
                gr.Examples(
                    examples=sorted(str(p) for p in EXAMPLES_DIR.iterdir()
                                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                    inputs=image,
                )

        with gr.Column():
            banner = gr.HTML()
            with gr.Row():
                with gr.Column(visible=False) as gemini_col:
                    gr.Markdown("### Gemini's opinion")
                    gemini_out = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### Image model (ResNet18), top 3")
                    resnet_out = gr.Label(num_top_classes=3, show_label=False)
            treatment = gr.Markdown()

    gr.Markdown(PROTOTYPE_NOTICE)

    submit.click(
        run_assessment,
        inputs=[image, *checkboxes],
        outputs=[banner, gemini_col, gemini_out, resnet_out, treatment],
    )


if __name__ == "__main__":
    demo.launch()
