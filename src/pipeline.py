import os

from .ocr import extract_text
from .rules import classify_text
from .utils import save_result
from .vision import detect_id_card
from .cnn_predict import predict_image


# ===============================
# Hyperparamètres simples
# ===============================
OCR_WEIGHT = 0.5
CNN_WEIGHT = 0.3
VISION_WEIGHT = 0.25

CNN_CONF_THRESHOLD = 0.6
VISION_CONF_THRESHOLD = 0.7
UNKNOWN_THRESHOLD = 0.2


def process_folder(input_dir, output_dir):
    """
    Pipeline final stable (V5) :

    1. OCR + règles (baseline dominante)
    2. Vision heuristique (priorité CIN)
    3. CNN finetuné (renfort image)
    4. Fusion pondérée simple
    5. unknown uniquement si signal trop faible
    """

    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            path = os.path.join(root, file)

            # =========================
            # 1. OCR + règles métier
            # =========================
            text = extract_text(path)
            ocr_label, ocr_score = classify_text(text)

            scores = {
                ocr_label: ocr_score * OCR_WEIGHT
            }

            # =========================
            # 2. Vision heuristique (CIN)
            # =========================
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                vision_score = detect_id_card(path)

                if vision_score >= VISION_CONF_THRESHOLD:
                    scores["piece_identite"] = max(
                        scores.get("piece_identite", 0),
                        vision_score * VISION_WEIGHT
                    )

            # =========================
            # 3. CNN finetuné
            # =========================
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    cnn_label, cnn_score = predict_image(path)

                    if cnn_score >= CNN_CONF_THRESHOLD:
                        scores[cnn_label] = max(
                            scores.get(cnn_label, 0),
                            cnn_score * CNN_WEIGHT
                        )

                except Exception:
                    pass  # sécurité

            # =========================
            # 4. Décision finale
            # =========================
            if not scores:
                final_label = "unknown"
                final_score = 0.0
            else:
                final_label = max(scores, key=scores.get)
                final_score = round(scores[final_label], 2)

                # sécurité : OCR tranche si score trop faible
                if final_score < UNKNOWN_THRESHOLD:
                    final_label = ocr_label
                    final_score = round(ocr_score, 2)

            # =========================
            # 5. Sauvegarde
            # =========================
            save_result(file, final_label, final_score, output_dir)
