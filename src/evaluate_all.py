import os, json
from sklearn.metrics import classification_report, confusion_matrix
from .ocr import extract_text, split_text_lang
from .rules import classify_text
from .nlp_predict import predict_text
from .cnn_predict import predict_image

RAW_DIR = 'data/raw'
RESULTS_DIR = 'results'

def gather_true_pred(pred_func):
    y_true = []
    y_pred = []
    for label in os.listdir(RAW_DIR):
        label_dir = os.path.join(RAW_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for f in os.listdir(label_dir):
            path = os.path.join(label_dir, f)
            pred_label, pred_score = pred_func(path)
            if pred_label == 'unknown':
                continue
            y_true.append(label)
            y_pred.append(pred_label)
    return y_true, y_pred

def pred_ocr(path):
    text = extract_text(path)
    return classify_text(text)

def pred_nlp(path):
    text = extract_text(path)
    fr, ar = split_text_lang(text)
    return predict_text(fr if fr.strip() else text)

def pred_cnn(path):
    if path.lower().endswith(('.jpg', '.jpeg', '.png')):
        return predict_image(path)
    return ('unknown', 0.0)

def pred_fusion(path):
    text = extract_text(path)
    fr, ar = split_text_lang(text)
    ocr_label, ocr_score = classify_text(text)
    nlp_label, nlp_score = predict_text(fr if fr.strip() else text)
    cnn_label, cnn_score = ('unknown', 0.0)
    if path.lower().endswith(('.jpg', '.jpeg', '.png')):
        cnn_label, cnn_score = predict_image(path)
    if nlp_score >= 0.5:
        return nlp_label, nlp_score
    if cnn_score >= 0.6:
        return cnn_label, cnn_score
    return ocr_label, ocr_score

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    for name, func in [('OCR', pred_ocr), ('NLP', pred_nlp), ('CNN', pred_cnn), ('FUSION', pred_fusion)]:
        y_true, y_pred = gather_true_pred(func)
        print('---', name, '---')
        if not y_true:
            print('No predictions for', name)
            continue
        print(classification_report(y_true, y_pred, zero_division=0))
        print('Confusion:')
        print(confusion_matrix(y_true, y_pred))
