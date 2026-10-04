"""DermaAssist web app. Start with: python app.py

Pipeline: predict (ResNet18) -> assess (Gemini) -> floor (urgency rules) -> render.
Nothing is saved to disk and inputs are never logged.
"""

import html
import json
from pathlib import Path
from urllib.parse import quote_plus

import gradio as gr

from clinics import ClinicSearchError, find_nearby, geocode

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
        return render_empty("Upload a photo of the skin spot first."), *NO_TREATMENT

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
        # No model opinions: ResNet18 names a lesion even for a photo of a wall.
        return render_report(final_level, gemini, rule, None), *NO_TREATMENT

    treatment = (gemini or {}).get("treatment_info") or ""
    return render_report(final_level, gemini, rule, probs), gr.update(visible=bool(treatment)), treatment


# --- Rendering ---

# Fixed message and colour per level. "Nothing flagged" is deliberately not green.
LEVEL_STYLE = {
    PROMPT_REVIEW: ("#c5372c", "See a doctor soon."),
    ROUTINE_REVIEW: ("#b98200", "Book a skin check."),
    NOTHING_FLAGGED: ("#5f6b7a", "Nothing was flagged. This is not an all-clear. "
                                 "See a doctor if the spot changes or worries you."),
    CANNOT_ASSESS: ("#8a8f98", "The tool cannot judge this image. "
                               "Retake it, or see a doctor if concerned."),
}

# Lowest to highest, as drawn on the urgency scale. Cannot assess sits outside it.
SCALE = [NOTHING_FLAGGED, ROUTINE_REVIEW, PROMPT_REVIEW]

CONFIDENCE_WIDTH = {"low": 33, "medium": 66, "high": 100}

# Inline SVG icons (Lucide-style strokes, coloured by currentColor).
_SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="{w}" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{d}</svg>')
ICONS = {
    "scan": _SVG.format(w=2, d='<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
                              '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
                              '<line x1="7" y1="12" x2="17" y2="12"/>'),
    "lock": _SVG.format(w=1.8, d='<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>'),
    "shield": _SVG.format(w=1.8, d='<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>'),
    "check": _SVG.format(w=1.8, d='<polyline points="20 6 9 17 4 12"/>'),
    "chevron": _SVG.format(w=1.8, d='<polyline points="9 18 15 12 9 6"/>'),
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


REPORT_HEAD = ('<div class="da-eyebrow">Analysis report</div>'
               '<h2 class="da-title">Assessment</h2>')


def render_empty(note=None):
    rows = "".join(
        f'<div class="da-map-row"><span class="da-letter">{i}</span>'
        f'<div class="da-map-main"><div class="da-map-line"><span>—</span><span class="da-val">— %</span></div>'
        f'<div class="da-under" style="width:18%"></div></div></div>'
        for i in (1, 2, 3)
    )
    note_html = f'<p class="da-warn">{esc(note)}</p>' if note else ""
    return (
        f'<div class="da-report">{REPORT_HEAD}'
        '<section class="da-sec"><div class="da-mono">Suggested urgency</div>'
        '<div class="da-level da-placeholder">Awaiting image</div>'
        '<p class="da-sub">Upload a close-up, tick any changes, then click Assess.</p>'
        f'{note_html}</section>'
        '<section class="da-sec"><div class="da-mono">Gemini confidence</div>'
        '<div class="da-big-dash">—<span>%</span></div><div class="da-meter"><div style="width:0"></div></div></section>'
        f'<section class="da-sec"><div class="da-sec-title">Image model · top 3</div>{rows}</section>'
        '</div>'
    )


def render_scale(level):
    if level not in SCALE:
        return ""
    segments = "".join(
        f'<div class="da-seg{" on" if name == level else ""}" style="--seg:{LEVEL_STYLE[name][0]}">'
        f'<i></i><span>{esc(name)}</span></div>'
        for name in SCALE
    )
    return f'<div class="da-scale" role="img" aria-label="Urgency: {esc(level)}">{segments}</div>'


def render_gemini(gemini):
    if gemini is None:
        return ('<section class="da-sec"><div class="da-mono">Gemini assessment</div>'
                '<p class="da-sub da-muted">AI assessment unavailable. '
                'Showing the image model and safety rules only.</p></section>')
    confidence = (gemini.get("confidence") or "").lower()
    return (
        '<section class="da-sec">'
        '<div class="da-row-top"><div class="da-mono">Gemini assessment</div>'
        f'<span class="da-badge">{esc(confidence or "unknown")}<br>confidence</span></div>'
        f'<div class="da-dx">{esc(gemini.get("diagnosis") or "Not given")}</div>'
        f'<p class="da-sub">{esc(gemini.get("visible_features") or "No visible features given.")}</p>'
        f'<div class="da-meter"><div style="width:{CONFIDENCE_WIDTH.get(confidence, 0)}%"></div></div>'
        '</section>'
    )


def render_resnet(gemini, probs):
    gemini_class = diagnosis_to_class(gemini.get("diagnosis")) if gemini else None
    resnet_top = max(probs, key=probs.get)
    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    rows = []
    for rank, (code, p) in enumerate(top3, start=1):
        match = code == gemini_class
        rows.append(
            f'<div class="da-map-row{" match" if match else ""}"><span class="da-letter">{rank}</span>'
            f'<div class="da-map-main"><div class="da-map-line"><span>{esc(CLASS_NAMES.get(code, code))}</span>'
            f'<span class="da-val">{round(p * 100)}%</span></div>'
            f'<div class="da-under" style="width:{max(round(p * 100), 4)}%"></div></div>'
            f'<span class="da-check" title="{"Matches Gemini" if match else ""}">{ICONS["check"] if match else ""}</span></div>'
        )

    if gemini is None:
        agree = ""
    elif gemini_class is None:
        agree = ('<p class="da-agree">Gemini\'s diagnosis is not one of the seven conditions '
                 'the image model knows, so the two cannot be compared directly.</p>')
    elif gemini_class != resnet_top:
        agree = (f'<p class="da-agree warn"><strong>The two models disagree.</strong> '
                 f'The image model\'s top answer is {esc(CLASS_NAMES[resnet_top])}.</p>')
    else:
        agree = '<p class="da-agree">Both models point to the same condition.</p>'

    return (f'<section class="da-sec"><div class="da-sec-title">Image model · ResNet18 top 3</div>'
            f'{"".join(rows)}{agree}</section>')


def render_report(level, gemini, rule, probs):
    colour, message = LEVEL_STYLE.get(level, LEVEL_STYLE[CANNOT_ASSESS])
    badge = '<span class="da-badge lvl">Raised by<br>safety rule</span>' if rule else ""
    issue = ""
    if level == CANNOT_ASSESS and gemini and gemini.get("image_issue"):
        issue = f'<p class="da-warn">{esc(gemini["image_issue"])}</p>'

    parts = [
        '<section class="da-sec">'
        f'<div class="da-row-top"><div class="da-mono">Suggested urgency</div>{badge}</div>'
        f'<div class="da-level">{esc(level)}</div><p class="da-sub">{esc(message)}</p>'
        f'{render_scale(level)}{issue}</section>'
    ]
    if probs is not None:
        parts.append(render_gemini(gemini))
        parts.append(render_resnet(gemini, probs))

    reasons = (gemini or {}).get("urgency_reasons") or []
    if not isinstance(reasons, list):
        reasons = [reasons]
    items = [f"<li>{esc(r)}</li>" for r in reasons]
    if rule:
        items.append(f'<li class="da-rule"><strong>Raised by safety rule:</strong> {esc(rule)}</li>')
    if items:
        parts.append(f'<div class="da-next"><div class="da-mono">Why this level</div>'
                     f'<ul>{"".join(items)}</ul></div>')

    return f'<div class="da-report" style="--lvl:{colour}">{REPORT_HEAD}{"".join(parts)}</div>'


# --- Find care ---

CARE_NOTICE = ("Your search location is sent to OpenStreetMap, rounded to about 1 km. Nothing is saved. "
               "Listings come from OpenStreetMap and may be incomplete or out of date, so call ahead.")

# Leaflet map in a sandboxed iframe: gr.HTML does not run scripts, an iframe srcdoc does.
MAP_DOC = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
html, body, #map { height: 100%; margin: 0; font-family: "IBM Plex Sans", system-ui, sans-serif; }
.pin { width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; color: #fff;
  font: 600 12px/1 system-ui, sans-serif; border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0,0,0,.35); }
.pin.derm { background: #4c7748; } .pin.other { background: #6b7280; }
</style></head><body><div id="map"></div><script>
const data = __DATA__;
const map = L.map("map");
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
const here = [data.lat, data.lon];
L.circleMarker(here, { radius: 7, color: "#fff", weight: 2, fillColor: "#2563eb", fillOpacity: 1 })
  .addTo(map).bindPopup("Search location");
const bounds = [here];
data.places.forEach((p, i) => {
  const icon = L.divIcon({ className: "", iconSize: [26, 26],
    html: `<div class="pin ${p.kind === "Dermatology" ? "derm" : "other"}">${i + 1}</div>` });
  const popup = document.createElement("div");
  popup.innerHTML = "<b></b><br><span></span>";
  popup.querySelector("b").textContent = p.name;
  popup.querySelector("span").textContent = p.kind + " · " + p.km.toFixed(1) + " km";
  L.marker([p.lat, p.lon], { icon }).addTo(map).bindPopup(popup);
  bounds.push([p.lat, p.lon]);
});
if (bounds.length > 1) map.fitBounds(bounds, { padding: [30, 30], maxZoom: 15 });
else map.setView(here, 13);
</script></body></html>"""


def directions_url(place):
    return f'https://www.google.com/maps/dir/?api=1&destination={place["lat"]},{place["lon"]}'


def render_care_empty(note=None):
    note_html = f'<p class="da-warn">{esc(note)}</p>' if note else ""
    return ('<div class="da-care-empty">'
            f'{note_html}<p class="da-sub">Type a city, suburb or postcode, or use your location, '
            'to see dermatologists and clinics near you.</p></div>')


def render_care(lat, lon, label, places, query):
    if label == "your location":
        google = f"https://www.google.com/maps/search/dermatologist/@{lat:.2f},{lon:.2f},13z"
    else:
        google = "https://www.google.com/maps/search/" + quote_plus(f"dermatologist near {query}")

    data = json.dumps({"lat": lat, "lon": lon, "places": places}).replace("</", "<\\/")
    iframe = (f'<iframe class="da-care-map" title="Map of nearby dermatologists and clinics" '
              f'sandbox="allow-scripts allow-popups" '
              f'srcdoc="{html.escape(MAP_DOC.replace("__DATA__", data), quote=True)}"></iframe>')

    if not places:
        listing = ('<p class="da-sub">No dermatologists or clinics are listed on OpenStreetMap near '
                   f'{esc(label)}. Try a nearby city, or search Google Maps below.</p>')
    else:
        derm_count = sum(p["kind"] == "Dermatology" for p in places)
        summary = (f"{derm_count} dermatology listing{'s' if derm_count != 1 else ''} within 25 km"
                   if derm_count else "No dermatology listings nearby, showing the nearest clinics. "
                                       "A GP can refer you to a dermatologist.")
        items = []
        for i, p in enumerate(places, start=1):
            derm = p["kind"] == "Dermatology"
            links = [f'<a href="{esc(directions_url(p))}" target="_blank" rel="noopener">Directions</a>']
            if p["phone"]:
                links.append(f'<a href="tel:{esc(p["phone"])}">{esc(p["phone"])}</a>')
            if p["website"]:
                links.append(f'<a href="{esc(p["website"])}" target="_blank" rel="noopener">Website</a>')
            items.append(
                f'<li class="da-care-item"><span class="da-care-num{" derm" if derm else ""}">{i}</span>'
                f'<div class="da-map-main"><div class="da-map-line"><b>{esc(p["name"])}</b>'
                f'<span class="da-val">{p["km"]:.1f} km</span></div>'
                f'<div class="da-care-meta"><span class="da-care-tag{" derm" if derm else ""}">{esc(p["kind"])}</span>'
                f'{esc(p["address"])}</div>'
                f'<div class="da-care-links">{" · ".join(links)}</div></div></li>'
            )
        listing = f'<p class="da-agree">{esc(summary)}</p><ol class="da-care-list">{"".join(items)}</ol>'

    return (f'<div class="da-care-grid">{iframe}<div>'
            f'<div class="da-mono">Near {esc(label)}</div>{listing}'
            f'<p class="da-care-more"><a href="{esc(google)}" target="_blank" rel="noopener">'
            'Search dermatologists on Google Maps ↗</a></p></div></div>')


def find_care(query):
    query = (query or "").strip()
    if not query:
        return render_care_empty("Enter a place, or allow location access.")
    try:
        lat, lon, label = geocode(query)
        places = find_nearby(lat, lon)
    except ClinicSearchError as e:
        return render_care_empty(str(e))
    return render_care(lat, lon, label, places, query)


# --- UI ---

HEADER = f"""
<header class="da-top">
  <div class="da-brand">
    <div class="da-mark">{ICONS["scan"]}</div>
    <div>
      <div class="da-name"> Dermassist <span class="da-ver">Research prototype</span></div>
      <div class="da-tagline">Skin spot triage workspace</div>
    </div>
  </div>
  <div class="da-top-right">
    <div class="da-meta"><div>Models</div><div class="da-meta-val">Gemini · ResNet18</div></div>
    <div class="da-vline"></div>
    <div class="da-private">{ICONS["lock"]}<span>Private session</span></div>
  </div>
</header>
"""

PROTOCOL = f"""
<div class="da-protocol">
  <div class="da-eyebrow">Screening protocol</div>
  <h1 class="da-headline">How soon should it be seen?</h1>
  <p class="da-lede">Two AI opinions and a set of safety rules suggest how urgently a doctor
  should look at a skin spot.</p>
  <hr>
  <div class="da-eyebrow">How it works</div>
  <ol class="da-steps">
    <li><span class="da-num">01</span><div><b>Capture</b><p>Use a close, well-lit, in-focus photo.</p></div></li>
    <li><span class="da-num">02</span><div><b>Answer</b><p>Tick any changes you have noticed.</p></div></li>
    <li><span class="da-num">03</span><div><b>Review</b><p>Read the urgency level and why.</p></div></li>
  </ol>
  <hr>
  <div class="da-aid">
    <div class="da-aid-head">{ICONS["shield"]}<b>Screening aid only</b></div>
    <p>{PROTOTYPE_NOTICE}</p>
    <p>{UPLOAD_NOTICE}</p>
  </div>
</div>
"""

CAPTURE_HEAD = """
<div class="da-panel-head">
  <div><div class="da-eyebrow">Step 1 · Capture</div><h2 class="da-title">Lesion capture</h2></div>
  <div class="da-status"><i></i>Nothing saved</div>
</div>
"""

# Gradio turns a leading "# " line into the placeholder heading, under its own upload icon.
UPLOAD_PLACEHOLDER = ("# Place image in the frame\n\n"
                      "Drop a close-up photo here, or click to choose a JPEG or PNG from your device.")

NO_TREATMENT = (gr.update(visible=False), "")

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.stone,
    radius_size="sm",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&display=swap">')

CSS = """
:root {
  --da-bg: #efede2; --da-panel: #f4f2e9; --da-line: #d9d5c4; --da-ink: #1f2a22; --da-muted: #5d675e;
  --da-label: #4e6a53; --da-sage: #5f8f5a; --da-sage-dark: #4c7748; --da-sage-soft: #dde6d3;
  --da-field: #d9e4d1; --da-grid: rgba(95, 143, 90, .16);
  --da-serif: "Source Serif 4", Georgia, serif; --da-mono: "IBM Plex Mono", ui-monospace, monospace;
}
.dark {
  --da-bg: #151915; --da-panel: #1b201b; --da-line: #2e372e; --da-ink: #e7ebe0; --da-muted: #a3ac9f;
  --da-label: #9cbf96; --da-sage: #4f7f4a; --da-sage-dark: #8fbd88; --da-sage-soft: #263224;
  --da-field: #1f2a1f; --da-grid: rgba(120, 169, 113, .13);
}
body, gradio-app, .gradio-container, .main, .wrap { background: var(--da-bg) !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; color: var(--da-ink); }
.gradio-container > .main > .wrap, .gradio-container .contain { padding: 0 !important; }
footer { display: none !important; }

.da-eyebrow { font-size: .74rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--da-ink); }
.da-mono, .da-sec-title { font-family: var(--da-mono); font-size: .72rem; letter-spacing: .04em;
  text-transform: uppercase; color: var(--da-label); }
.da-sec-title { font-family: inherit; font-weight: 600; letter-spacing: .12em; color: var(--da-ink); margin-bottom: 8px; }
.da-title { font-family: var(--da-serif) !important; font-weight: 400 !important; font-size: 2rem !important;
  margin: 2px 0 0 !important; color: var(--da-ink) !important; letter-spacing: -.01em; }

/* Header */
.da-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  padding: 22px clamp(16px, 6vw, 116px); border-bottom: 1px solid var(--da-line); background: var(--da-panel); }
.da-brand { display: flex; align-items: center; gap: 18px; }
.da-mark { width: 46px; height: 46px; display: grid; place-items: center; background: var(--da-sage); color: #fff; }
.da-mark svg { width: 22px; height: 22px; }
.da-name { font-family: var(--da-serif); font-size: 2rem; line-height: 1; color: var(--da-ink); }
.da-ver, .da-tagline, .da-meta, .da-private { font-family: var(--da-mono); font-size: .7rem; letter-spacing: .04em;
  text-transform: uppercase; color: var(--da-label); }
.da-ver { margin-left: 6px; vertical-align: middle; }
.da-tagline { margin-top: 6px; color: var(--da-ink); }
.da-top-right { display: flex; align-items: center; gap: 22px; }
.da-meta { text-align: right; line-height: 1.8; }
.da-meta-val { color: var(--da-ink); }
.da-vline { width: 1px; height: 38px; background: var(--da-line); }
.da-private { display: flex; align-items: center; gap: 8px; }
.da-private svg { width: 15px; height: 15px; }

/* Three-panel workspace */
#da-work { gap: 0 !important; align-items: stretch; flex-wrap: wrap; }
#da-work > .column { padding: 36px clamp(16px, 2.6vw, 38px) 40px !important; gap: 14px !important; }
#da-left { border-right: 1px solid var(--da-line); }
#da-right { border-left: 1px solid var(--da-line); }
#da-work .block, #da-work .form { background: transparent !important; border: 0 !important; box-shadow: none !important; }

/* Left: protocol */
.da-protocol hr { border: 0; border-top: 1px solid var(--da-line); margin: 30px 0 28px; }
.da-headline { font-family: var(--da-serif) !important; font-weight: 400 !important; font-size: clamp(2.2rem, 3.4vw, 3rem) !important;
  line-height: 1.02 !important; margin: 14px 0 22px !important; color: var(--da-ink) !important; letter-spacing: -.01em; }
.da-lede { color: var(--da-muted); line-height: 1.75; margin: 0; }
.da-steps { list-style: none; padding: 0; margin: 18px 0 0; }
.da-steps li { display: flex; gap: 22px; margin-bottom: 18px; }
.da-num { font-family: var(--da-mono); font-size: .72rem; color: var(--da-label); padding-top: 3px; }
.da-steps b { font-weight: 600; color: var(--da-ink); }
.da-steps p { margin: 4px 0 0; font-size: .86rem; color: var(--da-muted); }
.da-aid-head { display: flex; align-items: center; gap: 10px; color: var(--da-ink); font-size: .9rem; }
.da-aid-head svg { width: 17px; height: 17px; color: var(--da-sage); }
.da-aid p { font-size: .84rem; color: var(--da-muted); line-height: 1.6; margin: 10px 0 0; }
.da-aid p strong { color: var(--da-ink); font-weight: 500; }

/* Middle: capture */
.da-panel-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; margin-bottom: 8px; }
.da-status { display: flex; align-items: center; gap: 8px; font-family: var(--da-mono); font-size: .7rem;
  text-transform: uppercase; color: var(--da-label); padding-bottom: 8px; }
.da-status i { width: 7px; height: 7px; border-radius: 50%; background: var(--da-sage); }
#da-work #da-capture { position: relative; border: 1px solid color-mix(in srgb, var(--da-sage) 35%, transparent) !important;
  background-color: var(--da-field) !important;
  background-image: linear-gradient(var(--da-grid) 1px, transparent 1px), linear-gradient(90deg, var(--da-grid) 1px, transparent 1px) !important;
  background-size: 30px 30px !important; padding: 22px !important; }
#da-capture::after { content: ""; position: absolute; inset: 22px; pointer-events: none; z-index: 1;
  border: 1px solid color-mix(in srgb, var(--da-sage) 30%, transparent); --c: var(--da-sage-dark);
  background:
    linear-gradient(var(--c), var(--c)) top left / 26px 1.5px no-repeat, linear-gradient(var(--c), var(--c)) top left / 1.5px 26px no-repeat,
    linear-gradient(var(--c), var(--c)) top right / 26px 1.5px no-repeat, linear-gradient(var(--c), var(--c)) top right / 1.5px 26px no-repeat,
    linear-gradient(var(--c), var(--c)) bottom left / 26px 1.5px no-repeat, linear-gradient(var(--c), var(--c)) bottom left / 1.5px 26px no-repeat,
    linear-gradient(var(--c), var(--c)) bottom right / 26px 1.5px no-repeat, linear-gradient(var(--c), var(--c)) bottom right / 1.5px 26px no-repeat; }
#da-capture * { background-color: transparent !important; }
#da-capture .upload-container, #da-capture button { color: var(--da-ink) !important; }
#da-capture h2 { font-family: var(--da-serif) !important; font-weight: 400 !important; font-size: 1.85rem !important;
  color: var(--da-ink) !important; margin: 14px 0 6px !important; }
#da-capture p { max-width: 420px; margin: 0 auto !important; color: var(--da-muted) !important; line-height: 1.6; }
#da-capture .icon-wrap, #da-capture .upload-container svg { color: var(--da-label) !important; }

.da-label-row { margin-top: 6px; }
#da-q .wrap { gap: 0 !important; border: 1px solid var(--da-line); background: var(--da-panel); padding: 4px; width: fit-content; flex-wrap: wrap; }
#da-q label { background: transparent !important; border: 0 !important; box-shadow: none !important; border-radius: 2px !important;
  font-family: var(--da-mono); font-size: .72rem !important; text-transform: uppercase; letter-spacing: .03em;
  padding: 9px 13px !important; color: var(--da-ink) !important; cursor: pointer; }
#da-q label input { display: none !important; }
#da-q label.selected, #da-q label:has(input:checked) { background: var(--da-sage-soft) !important; color: var(--da-sage-dark) !important; }
#da-q label.selected::before, #da-q label:has(input:checked)::before { content: "✓ "; margin-right: 2px; }
#da-actions { gap: 10px !important; margin-top: 4px; }
#da-assess, #da-clear { border-radius: 2px !important; min-height: 46px; font-size: 1.02rem; box-shadow: 0 1px 2px rgba(0,0,0,.08); }
#da-assess { background: var(--da-sage) !important; color: #fff !important; border: 0 !important; }
#da-assess:hover { filter: brightness(.92); }
#da-clear { background: var(--da-panel) !important; color: var(--da-ink) !important; border: 1px solid var(--da-line) !important; }
#da-examples { margin-top: 6px; }
#da-examples .label, #da-examples span { font-family: var(--da-mono); font-size: .7rem; text-transform: uppercase; color: var(--da-label); }

/* Right: report */
.da-report { --lvl: var(--da-muted); }
.da-report > .da-title { padding-bottom: 22px; border-bottom: 1px solid var(--da-line); }
.da-sec { padding: 24px 0; border-bottom: 1px solid var(--da-line); animation: da-in .35s ease-out; }
.da-row-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.da-badge { font-family: var(--da-mono); font-size: .66rem; line-height: 1.4; text-transform: uppercase; letter-spacing: .04em;
  padding: 8px 12px; color: var(--da-sage-dark); background: var(--da-sage-soft);
  border: 1px solid color-mix(in srgb, var(--da-sage) 35%, transparent); }
.da-badge.lvl { color: var(--lvl); background: color-mix(in srgb, var(--lvl) 10%, transparent);
  border-color: color-mix(in srgb, var(--lvl) 35%, transparent); }
.da-level { font-family: var(--da-serif); font-size: 2.6rem; line-height: 1.05; color: var(--lvl); margin: 10px 0 6px; }
.da-level.da-placeholder { color: var(--da-muted); opacity: .6; }
.da-dx { font-family: var(--da-serif); font-size: 2.1rem; line-height: 1.1; color: var(--da-ink); margin: 6px 0 8px; }
.da-sub { margin: 0; color: var(--da-muted); line-height: 1.6; }
.da-muted { font-style: italic; }
.da-warn { margin: 14px 0 0; padding: 10px 12px; font-size: .9rem; background: color-mix(in srgb, var(--da-muted) 10%, transparent); }
.da-scale { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin-top: 18px; }
.da-seg i { display: block; height: 4px; background: color-mix(in srgb, var(--seg) 22%, transparent); }
.da-seg span { display: block; margin-top: 7px; font-family: var(--da-mono); font-size: .64rem; text-transform: uppercase; color: var(--da-muted); }
.da-seg.on i { background: var(--seg); height: 6px; margin-top: -1px; }
.da-seg.on span { color: var(--seg); font-weight: 600; }
.da-big-dash { font-size: 1.6rem; color: var(--da-ink); margin: 10px 0 14px; font-weight: 600; }
.da-big-dash span { font-family: var(--da-serif); font-weight: 400; color: var(--da-sage); margin-left: 4px; }
.da-meter { height: 4px; background: var(--da-sage-soft); margin-top: 18px; }
.da-meter div { height: 100%; background: var(--da-sage); animation: da-grow .7s ease-out; }
.da-map-row { display: flex; align-items: center; gap: 18px; padding: 16px 0 12px; border-bottom: 1px solid var(--da-line); }
.da-map-row:last-of-type { border-bottom: 0; }
.da-letter { font-family: var(--da-serif); font-size: 1.5rem; width: 22px; color: var(--da-label); }
.da-map-main { flex: 1; min-width: 0; }
.da-map-line { display: flex; justify-content: space-between; gap: 10px; font-size: .92rem; color: var(--da-ink); }
.da-val { font-variant-numeric: tabular-nums; color: var(--da-muted); }
.da-under { height: 2px; background: var(--da-sage); margin-top: 8px; animation: da-grow .7s ease-out; }
.da-map-row:not(:first-of-type) .da-under { background: color-mix(in srgb, var(--da-sage) 45%, transparent); }
.da-check { width: 16px; color: var(--da-sage); }
.da-check svg { width: 15px; height: 15px; }
.da-agree { margin: 12px 0 0; font-size: .88rem; color: var(--da-muted); line-height: 1.55; }
.da-agree.warn { color: var(--da-ink); padding: 10px 12px; background: color-mix(in srgb, #b98200 12%, transparent);
  border-left: 3px solid #b98200; }
.da-next { margin-top: 26px; padding: 20px 20px 18px; background: color-mix(in srgb, var(--da-sage) 12%, var(--da-panel));
  border-left: 3px solid var(--da-sage); }
.da-next ul { margin: 12px 0 0; padding-left: 18px; line-height: 1.6; color: var(--da-ink); }
.da-rule { color: var(--lvl); }
#da-work #da-treatment { margin-top: 10px; padding: 4px 14px !important; border: 1px solid var(--da-line) !important; border-radius: 0 !important; background: var(--da-panel) !important; }

/* Below: find care */
#da-care { padding: 36px clamp(16px, 6vw, 116px) 48px !important; border-top: 1px solid var(--da-line);
  background: var(--da-panel) !important; gap: 14px !important; }
#da-care .block, #da-care .form { background: transparent !important; border: 0 !important; box-shadow: none !important; }
#da-care-search { gap: 10px !important; align-items: stretch; max-width: 760px; }
#da-care-search textarea, #da-care-search input { border-radius: 2px !important; background: var(--da-bg) !important;
  border: 1px solid var(--da-line) !important; min-height: 46px; }
#da-locate, #da-find { border-radius: 2px !important; min-height: 46px; }
#da-find { background: var(--da-sage) !important; color: #fff !important; border: 0 !important; }
#da-locate { background: var(--da-bg) !important; color: var(--da-ink) !important; border: 1px solid var(--da-line) !important; }
.da-care-note { font-size: .82rem; color: var(--da-muted); margin: 0; max-width: 760px; line-height: 1.6; }
.da-care-empty { padding: 8px 0; }
.da-care-grid { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 28px; align-items: start;
  animation: da-in .35s ease-out; }
.da-care-map { width: 100%; height: 460px; border: 1px solid var(--da-line); background: var(--da-field); }
.da-care-list { list-style: none; padding: 0; margin: 10px 0 0; max-height: 420px; overflow-y: auto; }
.da-care-item { display: flex; gap: 14px; padding: 12px 0; border-bottom: 1px solid var(--da-line); }
.da-care-num { flex: none; width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center;
  font-size: .72rem; font-weight: 600; color: #fff; background: #6b7280; }
.da-care-num.derm { background: var(--da-sage-dark); }
.dark .da-care-num.derm { background: var(--da-sage); }
.da-care-meta { margin-top: 4px; font-size: .84rem; color: var(--da-muted); }
.da-care-tag { font-family: var(--da-mono); font-size: .64rem; text-transform: uppercase; letter-spacing: .04em;
  padding: 2px 6px; margin-right: 8px; border: 1px solid var(--da-line); color: var(--da-muted); }
.da-care-tag.derm { color: var(--da-sage-dark); background: var(--da-sage-soft); border-color: transparent; }
.da-care-links { margin-top: 6px; font-size: .84rem; }
.da-care-links a, .da-care-more a { color: var(--da-sage-dark); }
.da-care-more { margin: 14px 0 0; font-size: .88rem; }

@media (max-width: 900px) { .da-care-grid { grid-template-columns: 1fr; } .da-care-map { height: 340px; } }
@media (max-width: 1100px) { #da-left { border-right: 0; border-bottom: 1px solid var(--da-line); }
  #da-right { border-left: 0; border-top: 1px solid var(--da-line); } }
@keyframes da-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
@keyframes da-grow { from { width: 0; } }
"""


def busy():
    return gr.update(value="Assessing…", interactive=False)


def ready():
    return gr.update(value="Assess image", interactive=True)


def clear():
    return None, [], render_empty(), *NO_TREATMENT, gr.update(visible=False), render_care_empty()


def show_care(image):
    return gr.update(visible=image is not None)


# Browser geolocation, rounded to ~1 km. Keeps the typed place if access is refused.
LOCATE_JS = """
(current) => new Promise((resolve) => {
  if (!navigator.geolocation) { alert("Location is not available in this browser."); resolve(current); return; }
  navigator.geolocation.getCurrentPosition(
    (p) => resolve(p.coords.latitude.toFixed(2) + ", " + p.coords.longitude.toFixed(2)),
    () => { alert("Could not get your location. Type a city or postcode instead."); resolve(current); },
    { timeout: 10000, maximumAge: 600000 });
})
"""


with gr.Blocks(title="Dermassist", fill_width=True) as demo:
    gr.HTML(HEADER, padding=False)

    with gr.Row(elem_id="da-work", equal_height=False):
        with gr.Column(scale=3, min_width=260, elem_id="da-left"):
            gr.HTML(PROTOCOL, padding=False)

        with gr.Column(scale=6, min_width=340, elem_id="da-mid"):
            gr.HTML(CAPTURE_HEAD, padding=False)
            image = gr.Image(type="pil", sources=["upload"], show_label=False, height=440,
                             placeholder=UPLOAD_PLACEHOLDER, elem_id="da-capture")
            gr.HTML('<div class="da-mono da-label-row">Step 2 · Reported changes</div>', padding=False)
            questions = gr.CheckboxGroup(choices=list(QUESTIONS.values()), show_label=False, elem_id="da-q")
            with gr.Row(elem_id="da-actions"):
                clear_btn = gr.Button("Clear", scale=1, elem_id="da-clear")
                submit = gr.Button("Assess image", variant="primary", scale=3, elem_id="da-assess")
            if EXAMPLES_DIR.is_dir():
                with gr.Column(elem_id="da-examples"):
                    gr.Examples(
                        examples=sorted(str(p) for p in EXAMPLES_DIR.iterdir()
                                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                        inputs=image,
                        label="Or try an example",
                    )

        with gr.Column(scale=4, min_width=300, elem_id="da-right"):
            report = gr.HTML(render_empty(), padding=False)
            with gr.Accordion("How this is usually treated (general information)",
                              open=False, visible=False, elem_id="da-treatment") as treatment_box:
                treatment = gr.Markdown()

    with gr.Column(visible=False, elem_id="da-care") as care:
        gr.HTML('<div class="da-eyebrow">Step 3 · Find care</div>'
                '<h2 class="da-title">Dermatologists &amp; clinics nearby</h2>', padding=False)
        with gr.Row(elem_id="da-care-search"):
            place = gr.Textbox(show_label=False, placeholder="City, suburb or postcode", scale=4,
                               container=False, max_lines=1)
            locate_btn = gr.Button("Use my location", scale=2, elem_id="da-locate")
            find_btn = gr.Button("Search", scale=1, elem_id="da-find")
        gr.HTML(f'<p class="da-care-note">{esc(CARE_NOTICE)}</p>', padding=False)
        care_map = gr.HTML(render_care_empty(), padding=False)

    outputs = [report, treatment_box, treatment]
    (submit.click(busy, outputs=submit, queue=False)
        .then(run_assessment, inputs=[image, questions], outputs=outputs)
        .then(show_care, inputs=image, outputs=care, queue=False)
        .then(ready, outputs=submit, queue=False))
    clear_btn.click(clear, outputs=[image, questions, *outputs, care, care_map], queue=False)

    find_btn.click(find_care, inputs=place, outputs=care_map)
    place.submit(find_care, inputs=place, outputs=care_map)
    (locate_btn.click(None, inputs=place, outputs=place, js=LOCATE_JS)
        .then(find_care, inputs=place, outputs=care_map))


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD)
