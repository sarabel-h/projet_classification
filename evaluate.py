import os
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from torchvision import transforms
from transformers import CamembertTokenizer
from sklearn.model_selection import train_test_split # <--- Ajout important

# Imports de vos modules
from src.config import *
from src.ocr import extract_text_from_image, split_languages
from src.rules import get_rule_scores
from src.models_arch import CamembertClassifier, get_cnn_model
from src.fusion import global_fusion, predict_cnn_single, predict_nlp_single

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {DEVICE}")

def load_models():
    print(">>> Chargement des modèles...")
    # CNN
    cnn = get_cnn_model(len(CLASS_NAMES)).to(DEVICE)
    if os.path.exists(CNN_MODEL_PATH):
        cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
    cnn.eval()

    # NLP
    nlp = CamembertClassifier(len(CLASS_NAMES)).to(DEVICE)
    if os.path.exists(NLP_MODEL_PATH):
        nlp.load_state_dict(torch.load(NLP_MODEL_PATH, map_location=DEVICE))
    nlp.eval()

    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    return cnn, nlp, tokenizer

def evaluate_system():
    cnn_model, nlp_model, tokenizer = load_models()
    
    # Transform pour CNN
    trans_cnn = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    y_true = []
    y_pred_rules = []
    y_pred_cnn = []
    y_pred_nlp = []
    y_pred_fusion = []

    print("\n>>> Démarrage de l'évaluation sur le JEU DE TEST (20%)...")
    
    if not os.path.exists(DATA_PROCESSED):
        print("Erreur: Dossier data/processed_images introuvable.")
        return

    processed_count = 0

    for label_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(DATA_PROCESSED, cls_name)
        if not os.path.exists(cls_dir): continue
        
        all_files = os.listdir(cls_dir)
        
        # --- CORRECTION ICI ---
        # Si on a assez de fichiers, on refait le split pour isoler les 20% de test.
        # random_state=42 garantit que ce sont les MÊMES fichiers que ceux exclus pendant l'entraînement.
        if len(all_files) > 1:
            try:
                _, test_files = train_test_split(all_files, test_size=0.2, random_state=42, shuffle=True)
            except ValueError:
                # Si pas assez de données pour splitter, on prend tout (cas critique)
                test_files = all_files
        else:
            test_files = all_files

        print(f"   Classe {cls_name} : {len(test_files)} images de test.")

        for f in test_files:
            path = os.path.join(cls_dir, f)
            
            # --- 1. Vérité Terrain ---
            y_true.append(cls_name)
            
            # --- 2. Prédictions ---
            try:
                # A. Règles / OCR
                txt = extract_text_from_image(path)
                fr, _ = split_languages(txt)
                scores_rules, pred_rules = get_rule_scores(txt)
                y_pred_rules.append(pred_rules)

                # B. CNN
                from PIL import Image
                img = Image.open(path).convert('RGB')
                img_t = trans_cnn(img)
                scores_cnn = predict_cnn_single(cnn_model, img_t, DEVICE)
                pred_cnn = max(scores_cnn, key=scores_cnn.get)
                y_pred_cnn.append(pred_cnn)

                # C. NLP
                scores_nlp = predict_nlp_single(nlp_model, tokenizer, fr, DEVICE)
                pred_nlp = max(scores_nlp, key=scores_nlp.get)
                y_pred_nlp.append(pred_nlp)

                # D. FUSION
                scores_fusion = global_fusion(scores_rules, scores_nlp, scores_cnn)
                pred_fusion = max(scores_fusion, key=scores_fusion.get)
                
                if scores_fusion[pred_fusion] < THRESHOLD_UNKNOWN:
                    pred_fusion = "unknown"
                
                y_pred_fusion.append(pred_fusion)

            except Exception as e:
                # Fallback en cas d'erreur de lecture
                y_pred_rules.append("unknown")
                y_pred_cnn.append("unknown")
                y_pred_nlp.append("unknown")
                y_pred_fusion.append("unknown")

            processed_count += 1

    if processed_count == 0:
        print("Aucun fichier trouvé pour l'évaluation.")
        return

    print("\n\n>>> Génération des rapports...\n")

    def print_metrics(name, y_true, y_pred):
        print(f"=== RÉSULTATS : {name} ===")
        print(f"Accuracy Globale: {accuracy_score(y_true, y_pred):.2%}")
        print("-" * 60)
        print(classification_report(y_true, y_pred, zero_division=0))
        print("-" * 60)
        print("\n")

    print_metrics("APPROCHE 1: OCR + RÈGLES", y_true, y_pred_rules)
    print_metrics("APPROCHE 2: CNN", y_true, y_pred_cnn)
    print_metrics("APPROCHE 3: NLP", y_true, y_pred_nlp)
    print_metrics(">>> SYSTÈME FINAL (FUSION) <<<", y_true, y_pred_fusion)

if __name__ == "__main__":
    evaluate_system()