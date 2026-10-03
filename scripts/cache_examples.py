import hashlib
import json
import os
from PIL import Image
from gemini_assess import assess

def build_demo_cache(examples_dir="examples", output_cache_path="examples_cache.json"):
    """Pre-computes and caches Gemini assessments using image MD5 hashes."""
    if not os.path.exists(examples_dir):
        print(f"Directory '{examples_dir}' does not exist.")
        return

    cache = {}
    default_probs = {
        "akiec": 0.05, "bcc": 0.10, "bkl": 0.10,
        "df": 0.05, "mel": 0.15, "nv": 0.50, "vasc": 0.05
    }
    default_answers = {
        "grown": False, "changed": False,
        "bled": False, "itched": False, "hurt": False
    }

    for filename in os.listdir(examples_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            filepath = os.path.join(examples_dir, filename)
            try:
                img = Image.open(filepath)
                img_bytes = img.tobytes()
                img_hash = hashlib.md5(img_bytes).hexdigest()

                print(f"Caching Gemini response for {filename} (Hash: {img_hash[:8]}...)")
                res = assess(img, default_probs, default_answers)
                if res:
                    cache[img_hash] = res
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    with open(output_cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    print(f"Saved {len(cache)} responses to '{output_cache_path}'.")

if __name__ == "__main__":
    build_demo_cache()