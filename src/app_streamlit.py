import streamlit as st
from ocr import extract_text
from rules import classify_text

st.title("Offline Document Classifier")

uploaded = st.file_uploader("Uploader un document", type=["pdf", "png", "jpg", "jpeg"])
if uploaded:
    with open(uploaded.name, "wb") as f:
        f.write(uploaded.getbuffer())
    text = extract_text(uploaded.name)
    label, score = classify_text(text)
    st.success(f"Classe: {label} (confiance: {score})")
