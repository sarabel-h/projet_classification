import os
import torch
from transformers import CamembertTokenizer
from nlp_model import CamembertClassifier, CLASS_NAMES

MODEL_PATH = "models/camembert_classifier.pth"

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertClassifier()

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    except Exception:
        pass

model.eval()

def predict_text(text):
    # text should be preprocessed and be French-only for CamemBERT
    if not text or len(text.strip()) == 0:
        return "unknown", 0.0

    inputs = tokenizer(text[:1000], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs, dim=1)[0]

    score, idx = torch.max(probs, dim=0)
    return CLASS_NAMES[idx], round(score.item(), 2)
