
import streamlit as st
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import CamembertTokenizer
import shutil

from src.config import *
from src.preprocessing import convert_pdf_to_images
from src.ocr import extract_text_from_image, split_languages
from src.rules import get_rule_scores
from src.models_arch import CamembertClassifier, get_cnn_model
from src.fusion import global_fusion, predict_cnn_single, predict_nlp_single
from src.cin_detection import analyze_cin_structure

# --- Initialisation ---
st.set_page_config(page_title="Classificateur Doc Admin Offline", layout="wide")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CNN
    cnn = get_cnn_model(len(CLASS_NAMES))
    if os.path.exists(CNN_MODEL_PATH):
        cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    cnn.to(device)
    cnn.eval()
    
    # NLP
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    nlp = CamembertClassifier(len(CLASS_NAMES))
    if os.path.exists(NLP_MODEL_PATH):
        nlp.load_state_dict(torch.load(NLP_MODEL_PATH, map_location=device))
    nlp.to(device)
    nlp.eval()
    
    return cnn, nlp, tokenizer, device

cnn_model, nlp_model, tokenizer, device = load_models()

transform_cnn = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("📄 Classification de Documents (Multi-pages & Hybride)")
st.markdown("Système Offline : OCR + Règles + NLP (CamemBERT) + CNN (ResNet)")

uploaded_file = st.file_uploader("Choisissez un fichier (PDF ou Image)", type=["pdf", "jpg", "png", "jpeg"])

if "results" not in st.session_state:
    st.session_state.results = []

if uploaded_file is not None:
    if st.button("Lancer l'analyse"):
        st.session_state.results = [] # Reset
        temp_dir = "temp_upload"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Sauvegarder le fichier uploadé
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # 1. Conversion en images (Page par Page)
        st.info("Traitement du fichier et découpage des pages...")
        if file_path.lower().endswith('.pdf'):
            page_images = convert_pdf_to_images(file_path) # Retourne objets PIL
        else:
            page_images = [Image.open(file_path)]
            
        # 2. Analyse par page
        progress_bar = st.progress(0)
        
        for i, img in enumerate(page_images):
            # A. OCR & Règles
            text_full = extract_text_from_image(img)
            text_fr, text_ar = split_languages(text_full)
            ocr_scores, ocr_pred = get_rule_scores(text_full)
            
            # B. CNN Prediction
            img_tensor = transform_cnn(img.convert('RGB'))
            cnn_scores_dict = predict_cnn_single(cnn_model, img_tensor, device)
            
            # C. NLP Prediction
            nlp_scores_dict = predict_nlp_single(nlp_model, tokenizer, text_fr, device)
            
            # D. Fusion
            final_scores = global_fusion(ocr_scores, nlp_scores_dict, cnn_scores_dict)
            best_cls = max(final_scores, key=final_scores.get)
            confidence = final_scores[best_cls]
            
            # E. Gestion Unknown
            if confidence < THRESHOLD_UNKNOWN:
                best_cls = "unknown"
                
            # F. CIN Spécifique
            cin_detail = ""
            if best_cls == "piece_identite":
                side, side_conf = analyze_cin_structure(img)
                cin_detail = f"({side.upper()} - {int(side_conf*100)}%)"

            st.session_state.results.append({
                "page": i + 1,
                "image": img,
                "class": best_cls,
                "confidence": confidence,
                "cin_detail": cin_detail,
                "text": text_full,
                "details": {
                    "OCR Rules": ocr_scores,
                    "CNN Model": cnn_scores_dict,
                    "NLP Model": nlp_scores_dict
                }
            })
            progress_bar.progress((i + 1) / len(page_images))
            
        st.success("Analyse terminée !")

# --- Affichage des résultats ---
if st.session_state.results:
    st.divider()
    for res in st.session_state.results:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.image(res["image"], caption=f"Page {res['page']}", use_column_width=True)
            
        with c2:
            # En-tête coloré selon la confiance
            color = "green" if res["confidence"] > 0.7 else "orange" if res["confidence"] > 0.5 else "red"
            st.markdown(f"### Page {res['page']} : :{color}[{res['class']}] {res['cin_detail']}")
            st.progress(min(float(res["confidence"]), 1.0))
            st.caption(f"Confiance Globale : {res['confidence']:.2f}")
            
            with st.expander("🔍 Voir les détails (Scores par modèle & Texte)"):
                st.markdown("**Texte extrait (OCR) :**")
                st.text_area("Texte", res["text"], height=100, key=f"txt_{res['page']}")
                
                st.markdown("**Scores détaillés :**")
                df_scores = pd.DataFrame(res["details"])
                st.dataframe(df_scores.style.highlight_max(axis=0))

