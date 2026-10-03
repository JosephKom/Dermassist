import json
import os
from PIL import Image
from gemini_assess import assess

def cache_demo_responses(
    manifest_path="examples/examples_manifest.csv",
    examples_dir="examples",
    output_cache_path="examples_cache.json"
):
    """Pre-computes and caches Gemini responses for staged demo images."""
    if not os.path.exists(output_cache_path):
        cache = {}
    else:
        with open(output_cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    # Dummy baseline probabilities and patient responses for caching
    default_probs = {"mel": 0.15, "nv": 0.70, "bkl": 0.10}
    default_answers = {
        "has_grown": False,
        "has_changed": False,
        "has_bled": False,
        "is_itching": False,
        "is_painful": False
    }

    if not os.path.exists(examples_dir):
        print(f"Examples directory '{examples_dir}' not found. Skipping cache build.")
        return

    for filename in os.listdir(examples_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(examples_dir, filename)
            if filename in cache:
                print(f"Skipping already cached: {filename}")
                continue

            print(f"Caching Gemini response for: {filename}")
            try:
                img = Image.open(image_path)
                result = assess(img, default_probs, default_answers, cache_key=None)
                cache[filename] = result
            except Exception as e:
                print(f"Failed to cache {filename}: {e}")

    with open(output_cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    print(f"Successfully updated cache file '{output_cache_path}'.")

if __name__ == "__main__":
    cache_demo_responses()