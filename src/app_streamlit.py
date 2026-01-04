import streamlit as st
from .ocr import extract_text, split_text_lang
from .rules import classify_text
from .nlp_predict import predict_text
from .cnn_predict import predict_image
from .vision import detect_id_card

st.title('Offline Document Classifier (Demo)')
uploaded = st.file_uploader('Uploader un document', type=['pdf','png','jpg','jpeg'])
if uploaded:
    with open(uploaded.name, 'wb') as f:
        f.write(uploaded.getbuffer())
    path = uploaded.name
    text = extract_text(path)
    fr, ar = split_text_lang(text)
    ocr_label, ocr_score = classify_text(text)
    nlp_label, nlp_score = predict_text(fr if fr.strip() else text)
    cnn_label, cnn_score = ('unknown', 0.0)
    if path.lower().endswith(('.jpg', '.jpeg', '.png')):
        cnn_label, cnn_score = predict_image(path)
        vis_score = detect_id_card(path)
    st.write('OCR label:', ocr_label, ocr_score)
    st.write('NLP label:', nlp_label, nlp_score)
    st.write('CNN label:', cnn_label, cnn_score)
    if nlp_score >= 0.5:
        final = nlp_label
    elif cnn_score >= 0.6:
        final = cnn_label
    else:
        final = ocr_label
    st.success('Final: ' + final)
