import json
import os 
from typing import Any, Dict, Optional
from google import genai
from google.genai import types
from PIL import Image

def _get_client() -> Optional[genai.client]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_skin_lesion": {
            "type": "BOOLEAN",
            "description": "True if the image is a usable close-up of a skin lesion; False if non-skin, blank, out-of-focus, or unassessable."
        },
        "image_issue": {
            "type": "STRING",
            "description": "Explanation if is_skin_lesion is False; empty string if True."
        },
        "diagnosis": {
            "type": "STRING",
            "description": "Primary clinical assessment or probable condition name."
        },
        "confidence": {
            "type": "STRING",
            "description": "Level of diagnostic confidence: 'Low', 'Medium', or 'High'."
        },
        "visible_features": {
            "type": "STRING",
            "description": "Key visual observations regarding asymmetry, border, color, diameter, or surface details."
        },
        "urgency_level": {
            "type": "STRING",
            "enum": [
                "Cannot assess",
                "Prompt review",
                "Routine review",
                "Nothing flagged"
            ],
            "description": "Triage urgency level determined from visual assessment."
        },
        "urgency_reasons": {
            "type": "STRING",
            "description": "Clinical justification for the chosen urgency level."
        },
        "treatment_info": {
            "type": "STRING",
            "description": "General description of common management/treatment approaches. NO specific dosages or product names."
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

def _get_fallback_response(reason: str = "Gemini assessment unavailable") -> Dict[str, Any]:
    """Fallback payload returned when API fails, is blocked, or lacks an API key."""
    return {
        "is_skin_lesion": True,
        "image_issue": "",
        "diagnosis": "Unavailable (Fallback Mode)",
        "confidence": "N/A",
        "visible_features": "Assessment unavailable due to system fallback.",
        "urgency_level": "Routine review",
        "urgency_reasons": f"Secondary model assessment only ({reason}). Please consult a medical professional.",
        "treatment_info": "Consult a healthcare professional for diagnosis and treatment options."
    }

def _check_cache(cache_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Retrieves cached response from examples_cache.json if available."""
    cache_path = os.path.join(os.path.dirname(__file__), "examples_cache.json")
    if cache_key and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
                return cache.get(cache_key)
        except Exception:
            return None
    return None

def assess(
    image: Image.Image,
    probabilities: Dict[str, float],
    answers: Dict[str, bool],
    cache_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Makes a Gemini call to evaluate a skin lesion image alongside ResNet18 output and user history.
    
    Signature agreed with team:
        assess(image, probabilities, answers) -> dict
    """
    # 1. Check cached demo responses
    cached = _check_cache(cache_key)
    if cached:
        return cached

    # 2. Check client initialization
    client = _get_client()
    if not client:
        return _get_fallback_response("API key missing")

    # Format inputs for prompt context
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    probs_str = ", ".join([f"{cls}: {prob:.2%}" for cls, prob in sorted_probs[:3]])
    answers_str = ", ".join([f"{q}: {'Yes' if a else 'No'}" for q, a in answers.items()])

    prompt = f"""
You are a specialized dermatological triage assistant evaluating a patient-submitted image of a skin lesion.

[Context Provided]
- ResNet18 Top Probabilities: {probs_str}
- Patient Symptom Questionnaire: {answers_str}

[Assessment Guidelines]
1. First, check if the image is a valid, clear, close-up photograph of a skin lesion. If it is NOT a skin lesion (e.g., distant photo, non-skin object, extremely blurry, blank wall), set 'is_skin_lesion' to false, populate 'image_issue', set 'urgency_level' to 'Cannot assess', and provide advice to retake or see a doctor.
2. Select one of the four exact urgency levels:
   - 'Cannot assess' (image unusable)
   - 'Prompt review' (suspicious features or alarming symptoms present)
   - 'Routine review' (mildly atypical or uncertain features)
   - 'Nothing flagged' (benign appearance with no concerning flags)
3. Keep treatment information general and informative. DO NOT prescribe medications, name specific drug products, or state precise dosages.
4. Provide structured observations on color, border, symmetry, and visual characteristics.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GEMINI_RESPONSE_SCHEMA,
                temperature=0.1,
            )
        )
        if response.text:
            return json.loads(response.text)
        return _get_fallback_response("Empty response received")

    except Exception as e:
        print(f"[gemini_assess] Call failed: {e}")
        return _get_fallback_response(f"API Error: {str(e)}")