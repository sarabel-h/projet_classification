import os
from .ocr import extract_text, split_text_lang
from .rules import classify_text
from .utils import save_result
from .vision import detect_id_card
from .cnn_predict import predict_image
from .nlp_predict import predict_text

# weights
OCR_WEIGHT = 0.25
NLP_WEIGHT = 0.45
CNN_WEIGHT = 0.25
VISION_WEIGHT = 0.35

NLP_CONF_THRESHOLD = 0.45
CNN_CONF_THRESHOLD = 0.55
VISION_CONF_THRESHOLD = 0.7
UNKNOWN_THRESHOLD = 0.2

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            path = os.path.join(root, file)
            text = extract_text(path)
            fr_text, ar_text = split_text_lang(text)

            scores = {}
            weight_sum = 0.0

            # OCR rules baseline
            ocr_label, ocr_score = classify_text(text)
            if ocr_label != 'unknown':
                scores[ocr_label] = scores.get(ocr_label, 0.0) + ocr_score * OCR_WEIGHT
            weight_sum += OCR_WEIGHT

            # NLP (CamemBERT) on french text if available
            try:
                nlp_label, nlp_score = predict_text(fr_text if fr_text.strip() else text)
                if nlp_score >= NLP_CONF_THRESHOLD:
                    scores[nlp_label] = max(scores.get(nlp_label, 0.0), nlp_score * NLP_WEIGHT)
                weight_sum += NLP_WEIGHT
            except Exception:
                pass

            # vision heuristics (CIN)
            if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                vision_score = detect_id_card(path)
                if vision_score >= VISION_CONF_THRESHOLD:
                    scores['piece_identite'] = max(scores.get('piece_identite', 0.0), vision_score * VISION_WEIGHT)
                weight_sum += VISION_WEIGHT

            # CNN image
            if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    cnn_label, cnn_score = predict_image(path)
                    if cnn_score >= CNN_CONF_THRESHOLD:
                        scores[cnn_label] = max(scores.get(cnn_label, 0.0), cnn_score * CNN_WEIGHT)
                    weight_sum += CNN_WEIGHT
                except Exception:
                    pass

            # finalize: normalize by used weight sum
            if not scores:
                final_label = 'unknown'
                final_score = 0.0
            else:
                # select best label and normalize confidence
                best_label = max(scores, key=scores.get)
                raw = scores[best_label]
                # normalize into [0,1] by dividing by weight_sum
                final_conf = raw / (weight_sum if weight_sum>0 else 1.0)
                # boost if multiple modalities agree (simple)
                agree_count = 0
                try:
                    if ocr_label == best_label:
                        agree_count += 1
                    if 'nlp_label' in locals() and nlp_label == best_label:
                        agree_count += 1
                    if 'cnn_label' in locals() and cnn_label == best_label:
                        agree_count += 1
                    if path.lower().endswith(('.jpg', '.jpeg', '.png')) and detect_id_card(path) >= VISION_CONF_THRESHOLD and best_label == 'piece_identite':
                        agree_count += 1
                except Exception:
                    pass

                if agree_count >= 2:
                    final_conf = min(0.99, final_conf * 1.15)

                final_label = best_label if final_conf >= UNKNOWN_THRESHOLD else ocr_label
                final_score = round(min(0.99, final_conf), 2)

            save_result(file, final_label, final_score, output_dir)
