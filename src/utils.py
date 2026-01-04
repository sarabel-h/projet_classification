import json
import os

def save_result(filename, label, score, output_dir):
    result = {
        "file": filename,
        "predicted_class": label,
        "confidence": score
    }
    out = os.path.join(output_dir, filename + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
