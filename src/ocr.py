import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import re

def extract_text(path):
    """Extract text from PDF or image using OCR (French + Arabic). Returns full text."""
    text = ""

    if path.lower().endswith(".pdf"):
        images = convert_from_path(path)
        for img in images:
            page_text = pytesseract.image_to_string(
                img,
                lang="fra+ara",
                config="--psm 6"
            )
            text += "\n" + page_text
    else:
        img = Image.open(path)
        text = pytesseract.image_to_string(
            img,
            lang="fra+ara",
            config="--psm 6"
        )
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_text_lang(text):
    """Split text into French-like (non-Arabic) and Arabic segments."""
    fr_chars = []
    ar_chars = []
    for c in text:
        if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\uFB50' <= c <= '\uFDFF' or '\uFE70' <= c <= '\uFEFF':
            ar_chars.append(c)
        else:
            fr_chars.append(c)
    fr = ''.join(fr_chars).strip()
    ar = ''.join(ar_chars).strip()
    return fr, ar
