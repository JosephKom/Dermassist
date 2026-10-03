import csv
import glob
from gemini_assess import assess
from PIL import Image
from predictor import predict
from triage import floor

# Assumes sample images are named: <id>_<true_label>.jpg (e.g. ISIC_0024306_mel.jpg)
image_paths = sorted(glob.glob("examples/*.jpg"))[:20]

records = []

for path in image_paths:
    img = Image.open(path)
    true_label = path.split("_")[-1].split(".")[0]  # Extracts true label

    # 1. Run ResNet18
    probs = predict(img)
    top_resnet = max(probs, key=probs.get)

    # 2. Run Gemini
    default_answers = {
        "grown": False,
        "changed": False,
        "bled": False,
        "itched": False,
        "hurt": False,
    }
    gemini_res = assess(img, probs, default_answers)

    # 3. Apply floor
    triage_res = floor(gemini_res["urgency_level"], probs, default_answers)

    records.append(
        {
            "filename": path,
            "true_label": true_label,
            "resnet_top": top_resnet,
            "resnet_cancer_score": triage_res["cancer_score"],
            "gemini_diagnosis": gemini_res["diagnosis"],
            "gemini_urgency": gemini_res["urgency_level"],
            "final_urgency": triage_res["final_level"],
            "escalated": triage_res["escalated"],
            "agreement": (
                true_label.lower() in gemini_res["diagnosis"].lower()
            )
            and (top_resnet == true_label),
        }
    )

# Print Markdown Table to console / README
print(
    "| Image ID | True Label | ResNet18 Top | Cancer Score | Gemini Diagnosis | Gemini Urgency | Final Urgency | Esc? | Agree? |"
)
print(
    "|---|---|---|---|---|---|---|---|---|"
)
for r in records:
    print(
        f"| {r['filename']} | {r['true_label']} | {r['resnet_top']} | {r['resnet_cancer_score']:.2f} | "
        f"{r['gemini_diagnosis']} | {r['gemini_urgency']} | {r['final_urgency']} | {r['escalated']} | {r['agreement']} |"
    )