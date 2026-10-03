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
    "<strong>Research prototype. Not a medical device and not a diagnosis.</strong> "
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

def run_assessment(image, selected):
    if image is None:
        return render_message("Please upload a photo of the skin spot first."), *HIDDEN_DETAILS

    selected = selected or []
    answers = {key: question in selected for key, question in QUESTIONS.items()}
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

    if final_level == CANNOT_ASSESS:
        # Details hidden: ResNet18 names a lesion even for a photo of a wall.
        return render_result(final_level, gemini, rule), *HIDDEN_DETAILS

    treatment = render_treatment(gemini) if gemini else ""
    return (
        render_result(final_level, gemini, rule),
        gr.update(visible=True),
        render_opinions(gemini, probs),
        gr.update(visible=bool(treatment)),
        treatment,
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

# Lowest to highest, as drawn on the urgency scale. Cannot assess sits outside it.
SCALE = [NOTHING_FLAGGED, ROUTINE_REVIEW, PROMPT_REVIEW]

# Inline SVG icons (Lucide-style strokes, coloured by currentColor).
_SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{}</svg>')
ICONS = {
    PROMPT_REVIEW: _SVG.format('<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
                               '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'),
    ROUTINE_REVIEW: _SVG.format('<rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/>'
                                '<line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>'),
    NOTHING_FLAGGED: _SVG.format('<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/>'),
    CANNOT_ASSESS: _SVG.format('<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3"/>'
                               '<line x1="12" y1="17" x2="12.01" y2="17"/>'),
    "logo": _SVG.format('<path d="M12 3a9 9 0 1 0 9 9"/><circle cx="12" cy="12" r="4"/><path d="M16 3h5v5"/>'
                        '<line x1="21" y1="3" x2="15" y2="9"/>'),
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


def esc(value):
    return html.escape(str(value))


def render_message(text):
    return f'<div class="da-empty"><p>{esc(text)}</p></div>'


def render_scale(level):
    if level not in SCALE:
        return ""
    segments = "".join(
        f'<div class="da-seg{" on" if name == level else ""}" '
        f'style="--seg:{LEVEL_STYLE[name][0]}">{esc(name)}</div>'
        for name in SCALE
    )
    return f'<div class="da-scale" role="img" aria-label="Urgency: {esc(level)}">{segments}</div>'


def render_result(level, gemini, rule):
    colour, message = LEVEL_STYLE.get(level, LEVEL_STYLE[CANNOT_ASSESS])
    parts = [
        '<div class="da-result-head">'
        f'<div class="da-result-icon">{ICONS.get(level, ICONS[CANNOT_ASSESS])}</div>'
        f'<div><div class="da-eyebrow">Suggested urgency</div><h2>{esc(level)}</h2>'
        f'<p class="da-lead">{esc(message)}</p></div></div>',
        render_scale(level),
    ]

    if gemini is None:
        parts.append('<p class="da-note">AI assessment unavailable. '
                     'Showing the image model and safety rules only.</p>')
    elif level == CANNOT_ASSESS and gemini.get("image_issue"):
        parts.append(f'<p class="da-note">{esc(gemini["image_issue"])}</p>')

    reasons = (gemini or {}).get("urgency_reasons") or []
    if not isinstance(reasons, list):
        reasons = [reasons]
    items = [f"<li>{esc(r)}</li>" for r in reasons]
    if rule:
        items.append(f'<li class="da-rule"><strong>Raised by safety rule:</strong> {esc(rule)}</li>')
    if items:
        parts.append(f'<div class="da-subhead">Why</div><ul class="da-reasons">{"".join(items)}</ul>')

    return f'<div class="da-result" style="--lvl:{colour}">{"".join(parts)}</div>'


def render_bars(probs, highlight):
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    rows = []
    for code, p in top3:
        pct = round(p * 100)
        rows.append(
            f'<div class="da-bar{" match" if code == highlight else ""}">'
            f'<div class="da-bar-label"><span>{esc(CLASS_NAMES.get(code, code))}</span>'
            f'<span class="da-pct">{pct}%</span></div>'
            f'<div class="da-track"><div class="da-fill" style="width:{pct}%"></div></div></div>'
        )
    return "".join(rows)


def render_opinions(gemini, probs):
    resnet_top = max(probs, key=probs.get)
    gemini_class = diagnosis_to_class(gemini.get("diagnosis")) if gemini else None

    if gemini is None:
        gemini_body = '<p class="da-muted">Not available for this run.</p>'
    else:
        confidence = (gemini.get("confidence") or "").lower()
        filled = {"low": 1, "medium": 2, "high": 3}.get(confidence, 0)
        dots = "".join(f'<i class="{"on" if i < filled else ""}"></i>' for i in range(3))
        gemini_body = (
            f'<div class="da-dx">{esc(gemini.get("diagnosis") or "Not given")}</div>'
            f'<div class="da-conf"><span class="da-dots">{dots}</span>'
            f'{esc(confidence or "unknown")} confidence</div>'
            f'<div class="da-subhead">Visible features</div>'
            f'<p>{esc(gemini.get("visible_features") or "Not given")}</p>'
        )

    if gemini is None:
        agree = ""
    elif gemini_class is None:
        agree = ('<div class="da-agree info">Gemini\'s diagnosis is not one of the seven conditions '
                 'the image model knows, so the two cannot be compared directly.</div>')
    elif gemini_class != resnet_top:
        agree = (f'<div class="da-agree warn"><strong>The two models disagree.</strong> '
                 f'The image model\'s top answer is {esc(CLASS_NAMES[resnet_top])}.</div>')
    else:
        agree = '<div class="da-agree ok">Both models point to the same condition.</div>'

    return (
        '<div class="da-opinions">'
        f'<div class="da-op"><div class="da-op-head"><span class="da-tag">Gemini</span>Vision assessment</div>'
        f'{gemini_body}</div>'
        f'<div class="da-op"><div class="da-op-head"><span class="da-tag">ResNet18</span>Image model, top 3</div>'
        f'{render_bars(probs, gemini_class)}</div>'
        f'</div>{agree}'
    )


def render_treatment(gemini):
    return gemini.get("treatment_info") or ""


# --- UI ---

HERO = f"""
<div class="da-hero">
  <div class="da-logo">{ICONS["logo"]}</div>
  <div class="da-hero-text">
    <h1>DermaAssist</h1>
    <p>See how soon a skin spot should be checked by a doctor.</p>
  </div>
  <span class="da-pill">Research prototype</span>
</div>
<div class="da-notice">{PROTOTYPE_NOTICE}</div>
"""

TIPS = """
<ul class="da-tips">
  <li>Close up and in focus</li><li>Even, natural light</li><li>Spot in the centre</li>
</ul>
"""

EMPTY_STATE = """
<div class="da-empty">
  <div class="da-empty-title">Your result will appear here</div>
  <ol>
    <li>Upload a close-up photo of the spot</li>
    <li>Tick anything that applies to it</li>
    <li>Click <strong>Assess</strong></li>
  </ol>
</div>
"""

FOOTER = f'<div class="da-footer">Runs on your machine · {UPLOAD_NOTICE}</div>'

# Outputs that hide the details panel: (details column, opinions, treatment accordion, treatment).
HIDDEN_DETAILS = (gr.update(visible=False), "", gr.update(visible=False), "")

THEME = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1160px !important; margin: 0 auto !important; }
footer { display: none !important; }

.da-hero { display: flex; align-items: center; gap: 16px; padding: 8px 4px 4px; flex-wrap: wrap; }
.da-logo { width: 52px; height: 52px; border-radius: 14px; display: grid; place-items: center;
  color: #fff; background: linear-gradient(135deg, #14b8a6, #0f766e);
  box-shadow: 0 6px 18px rgba(13, 148, 136, .35); }
.da-logo svg { width: 28px; height: 28px; }
.da-hero-text { flex: 1; min-width: 200px; }
.da-hero h1 { margin: 0; font-size: 1.9rem; font-weight: 750; letter-spacing: -.02em; }
.da-hero p { margin: 2px 0 0; color: var(--body-text-color-subdued); }
.da-pill { font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em;
  padding: 6px 12px; border-radius: 999px; color: #0f766e;
  background: color-mix(in srgb, #14b8a6 16%, transparent); }
.dark .da-pill { color: #5eead4; }
.da-notice { margin: 12px 0 4px; padding: 10px 14px; border-radius: 10px; font-size: .9rem;
  border: 1px solid color-mix(in srgb, #e8a200 40%, transparent);
  background: color-mix(in srgb, #e8a200 10%, transparent); }

.da-card { border: 1px solid var(--border-color-primary) !important; border-radius: 16px !important;
  padding: 18px !important; background: var(--block-background-fill) !important;
  box-shadow: 0 1px 2px rgba(0,0,0,.04), 0 8px 24px rgba(0,0,0,.04); }
.da-step { display: flex; align-items: center; gap: 10px; font-weight: 650; font-size: 1.02rem; }
.da-step span { width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center;
  font-size: .85rem; color: #fff; background: #0d9488; }
.da-tips { display: flex; flex-wrap: wrap; gap: 6px; list-style: none; padding: 0; margin: 0; }
.da-tips li { font-size: .8rem; padding: 4px 10px; border-radius: 999px;
  background: var(--background-fill-secondary); color: var(--body-text-color-subdued); }
.da-tips li::before { content: "✓ "; color: #0d9488; font-weight: 700; }
#da-assess { font-size: 1.05rem; font-weight: 650; min-height: 48px; }

.da-empty { border: 2px dashed var(--border-color-primary); border-radius: 16px; padding: 28px;
  color: var(--body-text-color-subdued); }
.da-empty-title { font-size: 1.15rem; font-weight: 650; color: var(--body-text-color); margin-bottom: 8px; }
.da-empty ol { margin: 0; padding-left: 20px; line-height: 1.9; }

.da-result { border-radius: 16px; padding: 20px 22px; border: 1px solid color-mix(in srgb, var(--lvl) 35%, transparent);
  background: color-mix(in srgb, var(--lvl) 9%, var(--block-background-fill));
  box-shadow: inset 6px 0 0 var(--lvl); animation: da-in .35s ease-out; }
.da-result-head { display: flex; gap: 16px; align-items: flex-start; }
.da-result-icon { flex: none; width: 48px; height: 48px; border-radius: 12px; display: grid; place-items: center;
  color: #fff; background: var(--lvl); }
.da-result-icon svg { width: 26px; height: 26px; }
.da-eyebrow { font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
  color: var(--body-text-color-subdued); }
.da-result h2 { margin: 2px 0 4px !important; font-size: 1.6rem !important; font-weight: 750; color: var(--lvl) !important; }
.da-lead { margin: 0; font-weight: 550; }
.da-scale { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 18px 0 4px; }
.da-seg { text-align: center; font-size: .78rem; font-weight: 600; padding: 7px 4px; border-radius: 8px;
  color: var(--body-text-color-subdued); background: color-mix(in srgb, var(--seg) 12%, transparent); }
.da-seg.on { color: #fff; background: var(--seg); box-shadow: 0 4px 12px color-mix(in srgb, var(--seg) 45%, transparent); }
.da-note { margin: 14px 0 0; font-style: italic; color: var(--body-text-color-subdued); }
.da-subhead { margin: 16px 0 6px; font-size: .75rem; font-weight: 650; text-transform: uppercase;
  letter-spacing: .07em; color: var(--body-text-color-subdued); }
.da-reasons { margin: 0; padding-left: 20px; line-height: 1.6; }
.da-rule { color: var(--lvl); }

.da-opinions { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 720px) { .da-opinions { grid-template-columns: 1fr; } }
.da-op { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 16px;
  background: var(--block-background-fill); }
.da-op p { margin: 0; line-height: 1.55; }
.da-op-head { display: flex; align-items: center; gap: 8px; font-size: .85rem;
  color: var(--body-text-color-subdued); margin-bottom: 12px; }
.da-tag { font-size: .72rem; font-weight: 700; padding: 3px 8px; border-radius: 6px;
  color: #0f766e; background: color-mix(in srgb, #14b8a6 16%, transparent); }
.dark .da-tag { color: #5eead4; }
.da-dx { font-size: 1.2rem; font-weight: 700; line-height: 1.3; }
.da-conf { display: flex; align-items: center; gap: 8px; margin-top: 6px; font-size: .85rem;
  color: var(--body-text-color-subdued); text-transform: capitalize; }
.da-dots { display: inline-flex; gap: 3px; }
.da-dots i { width: 8px; height: 8px; border-radius: 50%; background: var(--border-color-primary); }
.da-dots i.on { background: #0d9488; }
.da-muted { color: var(--body-text-color-subdued); font-style: italic; }
.da-bar { margin-bottom: 12px; }
.da-bar-label { display: flex; justify-content: space-between; gap: 8px; font-size: .88rem; margin-bottom: 4px; }
.da-pct { font-variant-numeric: tabular-nums; font-weight: 650; }
.da-track { height: 8px; border-radius: 999px; background: var(--background-fill-secondary); overflow: hidden; }
.da-fill { height: 100%; border-radius: 999px; background: #94a3b8; animation: da-grow .6s ease-out; }
.da-bar:first-child .da-fill { background: #0d9488; }
.da-bar.match .da-bar-label span:first-child::after { content: "  · matches Gemini"; font-size: .75rem;
  color: #0d9488; font-weight: 600; }
.da-agree { margin-top: 12px; padding: 10px 14px; border-radius: 10px; font-size: .9rem; }
.da-agree.warn { background: color-mix(in srgb, #e8a200 14%, transparent); border: 1px solid color-mix(in srgb, #e8a200 40%, transparent); }
.da-agree.info, .da-agree.ok { background: var(--background-fill-secondary); }

.da-details { gap: 0 !important; }
.da-details .da-subhead { margin-top: 4px; }
.da-treat { border: 1px solid var(--border-color-primary) !important; border-radius: 14px !important; }
.da-footer { text-align: center; font-size: .8rem; color: var(--body-text-color-subdued); padding: 12px 0 4px; }

@keyframes da-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes da-grow { from { width: 0; } }
"""


def busy():
    return gr.update(value="Assessing…", interactive=False)


def ready():
    return gr.update(value="Assess", interactive=True)


def clear():
    return None, [], EMPTY_STATE, *HIDDEN_DETAILS


with gr.Blocks(title="DermaAssist") as demo:
    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="da-card"):
            gr.HTML('<div class="da-step"><span>1</span>Upload a close-up</div>')
            image = gr.Image(type="pil", sources=["upload"], show_label=False, height=300)
            gr.HTML(TIPS)
            if EXAMPLES_DIR.is_dir():
                gr.Examples(
                    examples=sorted(str(p) for p in EXAMPLES_DIR.iterdir()
                                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                    inputs=image,
                    label="Or try an example",
                )
            gr.HTML('<div class="da-step"><span>2</span>Tick anything that applies</div>')
            questions = gr.CheckboxGroup(choices=list(QUESTIONS.values()), show_label=False)
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                submit = gr.Button("Assess", variant="primary", scale=3, elem_id="da-assess")

        with gr.Column(scale=7):
            result = gr.HTML(EMPTY_STATE)
            with gr.Column(visible=False, elem_classes="da-details") as details:
                gr.HTML('<div class="da-subhead">Two opinions</div>')
                opinions = gr.HTML()
            with gr.Accordion("How this is usually treated (general information)",
                              open=False, visible=False, elem_classes="da-treat") as treatment_box:
                treatment = gr.Markdown()

    gr.HTML(FOOTER)

    outputs = [result, details, opinions, treatment_box, treatment]
    (submit.click(busy, outputs=submit, queue=False)
        .then(run_assessment, inputs=[image, questions], outputs=outputs)
        .then(ready, outputs=submit, queue=False))
    clear_btn.click(clear, outputs=[image, questions, *outputs], queue=False)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
