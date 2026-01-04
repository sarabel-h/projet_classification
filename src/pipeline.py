import os

from ocr import extract_text, split_text_lang
from rules import classify_text
from utils import save_result
from vision import detect_id_card
from cnn_predict import predict_image
from nlp_predict import predict_text


# ===============================
# Poids des modalités
# ===============================
OCR_WEIGHT = 0.25
NLP_WEIGHT = 0.45
CNN_WEIGHT = 0.25
VISION_WEIGHT = 0.35

# Seuils
NLP_CONF_THRESHOLD = 0.45
CNN_CONF_THRESHOLD = 0.55
VISION_CONF_THRESHOLD = 0.7
UNKNOWN_THRESHOLD = 0.45


def process_folder(input_dir, output_dir):
    """
    - OCR + règles (baseline)
    - NLP (CamemBERT) sur texte FR
    - Vision heuristique (CIN)
    - CNN image
    - Fusion additive + confidence relative
    """

    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            path = os.path.join(root, file)

            # =========================
            # 1. OCR
            # =========================
            text = extract_text(path)
            fr_text, ar_text = split_text_lang(text)

            scores = {}

            # =========================
            # 2. OCR rules (baseline)
            # =========================
            ocr_label, ocr_score = classify_text(text)
            if ocr_label != "unknown":
                scores[ocr_label] = scores.get(ocr_label, 0.0) + ocr_score * OCR_WEIGHT

            # =========================
            # 3. NLP (CamemBERT)
            # =========================
            try:
                nlp_input = fr_text if fr_text.strip() else text
                nlp_label, nlp_score = predict_text(nlp_input)

                if nlp_score >= NLP_CONF_THRESHOLD:
                    scores[nlp_label] = scores.get(nlp_label, 0.0) + nlp_score * NLP_WEIGHT
            except Exception:
                pass

            # =========================
            # 4. Vision heuristique (CIN)
            # =========================
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                vision_score = detect_id_card(path)
                if vision_score >= VISION_CONF_THRESHOLD:
                    scores["piece_identite"] = scores.get(
                        "piece_identite", 0.0
                    ) + vision_score * VISION_WEIGHT

            # =========================
            # 5. CNN image
            # =========================
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    cnn_label, cnn_score = predict_image(path)
                    if cnn_score >= CNN_CONF_THRESHOLD:
                        scores[cnn_label] = scores.get(
                            cnn_label, 0.0
                        ) + cnn_score * CNN_WEIGHT
                except Exception:
                    pass

            # =========================
            # 6. Décision finale
            # =========================
            if not scores:
                final_label = "unknown"
                final_score = 0.0
            else:
                best_label = max(scores, key=scores.get)
                best_score = scores[best_label]
                total_score = sum(scores.values())

                # Confidence relative (CORRECTE)
                final_conf = best_score / total_score if total_score > 0 else 0.0

                # Boost si plusieurs signaux convergent
                agree = 0
                if ocr_label == best_label:
                    agree += 1
                try:
                    if nlp_label == best_label:
                        agree += 1
                except:
                    pass
                try:
                    if cnn_label == best_label:
                        agree += 1
                except:
                    pass

                if agree >= 2:
                    final_conf *= 1.15

                final_conf = min(final_conf, 0.95)

                if final_conf < UNKNOWN_THRESHOLD:
                    final_label = "unknown"
                    final_score = round(final_conf, 2)
                else:
                    final_label = best_label
                    final_score = round(final_conf, 2)

            # =========================
            # 7. Sauvegarde
            # =========================
            save_result(file, final_label, final_score, output_dir)
