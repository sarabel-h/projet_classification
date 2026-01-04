import json, os
def save_result(filename, label, score, output_dir):
    result = {'file': filename, 'predicted_class': label, 'confidence': score}
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, filename + '.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
