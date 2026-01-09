
import pytesseract
import re

def extract_text_from_image(image_obj):
    """
    Applique Tesseract sur une image (PIL Object ou Path).
    Gère le multilingue (fra+ara).
    """
    try:
        # Configuration PSM 6 (Assume un bloc de texte uniforme) ou 3 (Auto)
        # On utilise fra+ara pour capturer les deux langues
        text = pytesseract.image_to_string(image_obj, lang='fra+ara', config='--psm 3')
        return text.strip()
    except Exception as e:
        print(f"Erreur OCR: {e}")
        return ""

def split_languages(text):
    """
    Sépare grossièrement le texte latin (Français) et Arabe.
    Utile car CamemBERT ne comprend pas l'arabe.
    """
    fr_chars = []
    ar_chars = []
    
    for char in text:
        # Plages Unicode Arabe basique
        if '\u0600' <= char <= '\u06FF': 
            ar_chars.append(char)
        else:
            fr_chars.append(char)
            
    return "".join(fr_chars).strip(), "".join(ar_chars).strip()
