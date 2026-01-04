import cv2
import numpy as np
import pytesseract
import json
import re
import unicodedata
import pypdfium2 as pdfium
from pathlib import Path

class DetecteurFacture:
    def __init__(self, chemin_gabarits="models/gabarits/gabarits_maroc.json"):
        self.config = self._charger_gabarits(chemin_gabarits)

    def _charger_gabarits(self, chemin):
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("facture_eau_electricite", {})
        except FileNotFoundError:
            return {}

    def _nettoyer_texte(self, texte):
        if not texte: return ""
        nfkd = unicodedata.normalize('NFKD', texte)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower()

    def _ocr_complet(self, image):
        # 1. Conversion Gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Amélioration Contraste
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
        
        # 3. Binarisation
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # 4. ASTUCE : Dilatation légère pour renforcer les traits fins (ex: le 'k' de kWh)
        kernel = np.ones((1, 1), np.uint8) 
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        return pytesseract.image_to_string(thresh, lang="fra+ara", config="--psm 6")

    def _detecter_type_fluide(self, texte_clean, texte_brut):
        categories = self.config.get("categories", {})
        scores = {}
        found_keywords = {} # Pour le debug

        for fluide, regles in categories.items():
            score = 0
            found_keywords[fluide] = []
            
            # Regex Unités (+50 pts)
            for pattern in regles.get("units_regex", []):
                if re.search(pattern, texte_clean):
                    score += 50
                    found_keywords[fluide].append(f"REGEX:{pattern}")
                    break 

            # Mots-clés (+10 ou +30 pts)
            for kw in regles.get("keywords", []):
                if kw in texte_clean or kw in texte_brut:
                    # Si c'est un mot fort (taxe, redevance), gros points
                    if any(x in kw for x in ["redevance", "fixe", "audiovisuel", "csave", "bav"]):
                        score += 30
                        found_keywords[fluide].append(f"FORT:{kw}")
                    else:
                        score += 10
                        found_keywords[fluide].append(f"KW:{kw}")

            scores[fluide] = score

        # SEUIL TRES BAS pour attraper les cas difficiles
        seuil = 15 
        
        s_elec = scores.get("electricite", 0)
        s_eau = scores.get("eau", 0)
        
        # Info debug pour l'utilisateur
        debug_msg = f"Scores: EAU={s_eau} ELEC={s_elec} | Trouvé: {found_keywords}"

        if s_elec >= seuil and s_eau >= seuil:
            return "Mixte (Eau + Électricité)", (s_elec + s_eau) / 2, debug_msg
        elif s_elec >= seuil:
            return "Électricité Seule", s_elec, debug_msg
        elif s_eau >= seuil:
            return "Eau Seule", s_eau, debug_msg
        else:
            return "Inconnu", 0, debug_msg

    def _detecter_fournisseur(self, texte_clean, texte_brut):
        providers = self.config.get("providers", {})
        for nom, mots_cles in providers.items():
            if any(m in texte_clean for m in mots_cles) or any(m in texte_brut for m in mots_cles):
                return nom
        return "Autre"

    def analyser(self, images_input):
        if not isinstance(images_input, list): images_input = [images_input]
        
        full_text_clean = ""
        full_text_brut = ""
        
        for img in images_input:
            if img is None: continue
            txt = self._ocr_complet(img)
            full_text_brut += txt + "\n"
            full_text_clean += self._nettoyer_texte(txt) + "\n"

        mots_generaux = self.config.get("general_keywords", [])
        is_facture = any(k in full_text_clean for k in mots_generaux) or \
                     any(k in full_text_brut for k in mots_generaux)
        
        if not is_facture:
            return {"valide": False, "message": "Non reconnu", "score": 0, "debug": "Pas de mots clés facture"}

        type_facture, score_type, debug_info = self._detecter_type_fluide(full_text_clean, full_text_brut)
        fournisseur = self._detecter_fournisseur(full_text_clean, full_text_brut)
        
        # Montant
        montants = re.findall(r"(?:total|payer|net|montant|مجموع|أداء).*?(\d+[.,]\d{2})", full_text_clean + full_text_brut, re.IGNORECASE)
        montant_final = "0.00"
        if montants:
            try: montant_final = f"{max([float(m.replace(',', '.')) for m in montants]):.2f}"
            except: pass

        return {
            "valide": True,
            "type_facture": type_facture,
            "fournisseur": fournisseur,
            "montant_total_prob": montant_final,
            "score": min(100, int(score_type + 20)),
            "debug": debug_info
        }

def pdf_vers_images(chemin_pdf):
    try:
        pdf = pdfium.PdfDocument(str(chemin_pdf))
        images = []
        for i in range(len(pdf)):
            images.append(cv2.cvtColor(np.array(pdf[i].render(scale=3).to_pil()), cv2.COLOR_RGB2BGR))
        return images
    except: return []