
import torch
import numpy as np
from src.config import CLASS_NAMES

def predict_cnn_single(model, img_tensor, device):
    """Retourne dict {classe: proba}"""
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

def predict_nlp_single(model, tokenizer, text, device):
    """Retourne dict {classe: proba}"""
    if not text.strip():
        return {c: 0.0 for c in CLASS_NAMES}
        
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

def global_fusion(ocr_scores, nlp_scores, cnn_scores):
    """
    Moyenne pondérée des scores.
    Poids ajustables selon la fiabilité observée.
    """
    # Poids
    W_OCR = 0.3
    W_NLP = 0.4
    W_CNN = 0.3
    
    final_scores = {}
    
    for cls in CLASS_NAMES:
        s_ocr = ocr_scores.get(cls, 0.0)
        s_nlp = nlp_scores.get(cls, 0.0)
        s_cnn = cnn_scores.get(cls, 0.0)
        
        # Logique spéciale: Si OCR détecte 'Assainissement' ou 'KWh' (mots très forts), on boost l'OCR
        # C'est géré implicitement par les exclusions dans rules.py qui mettent des scores négatifs
        
        score = (s_ocr * W_OCR) + (s_nlp * W_NLP) + (s_cnn * W_CNN)
        final_scores[cls] = score
        
    return final_scores
