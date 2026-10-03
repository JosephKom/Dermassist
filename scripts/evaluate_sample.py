"""Run the pipeline against the bundled example images."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gemini_assess import assess
from predictor import predict
from triage import floor


def main() -> None:
    image_paths = sorted((ROOT / "examples").rglob("*.jpg"))[:20]
    if not image_paths:
        raise SystemExit("No .jpg files found under examples/")

    answers = {
        "grown": False,
        "changed": False,
        "bled": False,
        "itched": False,
        "hurt": False,
    }

    print("| Image | ResNet18 top | Cancer score | Gemini urgency | Final urgency |")
    print("|---|---|---:|---|---|")
    for path in image_paths:
        with Image.open(path) as image:
            probabilities = predict(image)
            gemini = assess(image, probabilities, answers)

        level = gemini.get("urgency_level") if gemini else None
        final_level, _ = floor(level, probabilities, answers)
        top_class = max(probabilities, key=probabilities.get)
        cancer_score = probabilities["mel"] + probabilities["bcc"]
        print(
            f"| {path.relative_to(ROOT)} | {top_class} | {cancer_score:.2%} | "
            f"{level or 'Unavailable'} | {final_level} |"
        )


if __name__ == "__main__":
    main()
