import hashlib
import json
import os
from typing import Any, Dict, Optional
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# 1. Exact shared urgency enum strings
URGENCY_LEVELS = [
    "Cannot assess",
    "Prompt review",
    "Routine review",
    "Nothing flagged"
]

# 2. Structured output schema definition for Gemini
GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_skin_lesion": {
            "type": "BOOLEAN",
            "description": "True if the image is a usable close-up of a skin lesion; False if non-skin, distant photo, or unassessable."
        },
        "image_issue": {
            "type": "STRING",
            "nullable": True,
            "description": "Short explanation if is_skin_lesion is False; null or empty string if True."
        },
        "diagnosis": {
            "type": "STRING",
            "description": "Primary visual assessment or probable skin condition name."
        },
        "confidence": {
            "type": "STRING",
            "enum": ["low", "medium", "high"],
            "description": "Level of diagnostic confidence: 'low', 'medium', or 'high'."
        },
        "visible_features": {
            "type": "STRING",
            "description": "Key visual features (e.g., asymmetry, border irregularity, color variations, diameter)."
        },
        "urgency_level": {
            "type": "STRING",
            "enum": URGENCY_LEVELS,
            "description": "Exact triage urgency level matching one of the required four levels."
        },
        "urgency_reasons": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "List of concise reasons for assigning this urgency level."
        },
        "treatment_info": {
            "type": "STRING",
            "description": "General explanation of standard clinical management. NO specific dosages or product names."
        }
    },
    "required": [
        "is_skin_lesion",
        "image_issue",
        "diagnosis",
        "confidence",
        "visible_features",
        "urgency_level",
        "urgency_reasons",
        "treatment_info"
    ]
}

def _get_client() -> Optional[genai.Client]:
    """Retrieves Google GenAI SDK client if API key is present."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None

def _compute_image_hash(image: Image.Image) -> str:
    """Generates an MD5 fingerprint of image bytes for cache lookup."""
    try:
        img_bytes = image.tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    except Exception:
        return ""

def _check_cache(image: Image.Image) -> Optional[Dict[str, Any]]:
    """Internal cache lookup so app.py doesn't need to know about demo caching."""
    cache_path = os.path.join(os.path.dirname(__file__), "examples_cache.json")
    if not os.path.exists(cache_path):
        return None

    img_hash = _compute_image_hash(image)
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
            return cache.get(img_hash)
    except Exception:
        return None

def assess(
    image: Image.Image,
    probabilities: Dict[str, float],
    answers: Dict[str, bool]
) -> Optional[Dict[str, Any]]:
    """
    Evaluates a skin lesion image using Gemini 2.5 Flash.

    Contract Signature:
        assess(image, probabilities, answers) -> dict | None

    Returns:
        Dict matching shared specification when successful, or None on failure/blocked calls.
    """
    # Step A: Check cached demo replies internally
    cached_response = _check_cache(image)
    if cached_response:
        return cached_response

    # Step B: Check API client availability
    client = _get_client()
    if not client:
        print("[gemini_assess] Missing GEMINI_API_KEY or client initialization failed.")
        return None

    # Format inputs according to shared contract specification
    probs_formatted = ", ".join([f"{code}: {prob:.3f}" for code, prob in probabilities.items()])
    answers_formatted = ", ".join([
        f"Grown: {answers.get('grown', False)}",
        f"Changed: {answers.get('changed', False)}",
        f"Bled: {answers.get('bled', False)}",
        f"Itched: {answers.get('itched', False)}",
        f"Hurt: {answers.get('hurt', False)}"
    ])

    prompt = f"""
You are a dermatological assessment assistant evaluating a patient image.

[System Inputs]
- ResNet18 Class Probabilities: {{{probs_formatted}}}
- Patient History Answers: {{{answers_formatted}}}

[Instructions]
1. Scope Check: First check if the image is a clear close-up of a skin lesion.
   - If NOT a usable skin lesion (e.g., room photo, non-skin object, extremely blurry), set 'is_skin_lesion' to false, populate 'image_issue', and set 'urgency_level' to 'Cannot assess'.
2. Urgency Selection: Select EXACTLY one of the 4 allowed urgency levels:
   - "Cannot assess" (image unusable)
   - "Prompt review" (concerning symptoms or atypical/malignant appearance)
   - "Routine review" (mildly atypical or uncertain features requiring routine doctor review)
   - "Nothing flagged" (reassuring benign appearance with no suspicious history)
3. Reasons: Provide 'urgency_reasons' as a list of strings explaining why this urgency level was chosen.
4. Treatment Info: Provide general educational information on how this condition is typically evaluated or managed. Do NOT mention specific drug names, brand names, or medication dosages.
"""

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GEMINI_RESPONSE_SCHEMA,
                temperature=0.1,
            )
        )

        if not response.text:
            return None

        data = json.loads(response.text)

        # Ensure urgency_reasons is always formatted as a list[str]
        if isinstance(data.get("urgency_reasons"), str):
            data["urgency_reasons"] = [data["urgency_reasons"]]

        return data

    except Exception as e:
        print(f"[gemini_assess] API call failed: {e}")
        return None