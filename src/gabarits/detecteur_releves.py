import cv2
import numpy as np
import pytesseract
import json
import re
import pypdfium2 as pdfium 
from pathlib import Path

class DetecteurReleveBancaire:
    def __init__(self, chemin_gabarits="gabarits_maroc.json"):
        self.config = self._charger_gabarits(chemin_gabarits)

    def _charger_gabarits(self, chemin):
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("releve_bancaire_maroc", {})
        except FileNotFoundError:
            print(f"⚠️ Erreur configuration : Fichier {chemin} introuvable.")
            return {}

    def _extraire_zone(self, image, zone):
        h, w = image.shape[:2]
        x1, y1 = int(zone[0] * w), int(zone[1] * h)
        x2, y2 = int(zone[2] * w), int(zone[3] * h)
        return image[y1:y2, x1:x2]

    def _ocr_zone(self, image, zone=None):
        img_crop = self._extraire_zone(image, zone) if zone else image
        # Conversion niveau de gris
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        # Binarisation (Noir et Blanc pur) pour aider Tesseract
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return pytesseract.image_to_string(thresh, lang="fra", config="--psm 6").lower()

    def _detecter_banque(self, texte):
        banques = {
            "CIH BANK": ["cih", "cih bank"],
            "ATTIJARIWAFA BANK": ["attijari", "attijariwafa", "wafa"],
            "BANQUE POPULAIRE": ["banque populaire", "chaabi", "bcp", "assoukhour"],
            "AL BARID BANK": ["al barid", "barid bank", "poste maroc"],
            "BMCE": ["bmce", "bank of africa"],
            "CFG BANK": ["cfg bank", "cfg"],
            "SOCIETE GENERALE": ["societe generale", "sgmb"]
        }
        for nom_officiel, keywords in banques.items():
            if any(k in texte for k in keywords):
                return nom_officiel
        return None

    def _detecter_rib(self, texte):
        # Cherche 24 chiffres (nettoie les espaces/tirets automatiquement)
        pattern = r"\b(?:\d[\s\.-]*){24}\b"
        matches = re.findall(pattern, texte)
        for match in matches:
            rib_clean = re.sub(r"[^\d]", "", match)
            if len(rib_clean) == 24:
                return rib_clean
        return None

    def analyser(self, image_input):
        if image_input is None:
            return {"valide": False, "erreur": "Image vide"}

        score = 0
        details = {}
        features = self.config.get("features", [])
        full_text_buffer = ""

        # 1. Analyse Entête (Banque)
        zone_entete = next((f for f in features if f["nom"] == "entete_banque_logo"), None)
        if zone_entete:
            txt = self._ocr_zone(image_input, zone_entete["zone"])
            full_text_buffer += txt + " "
            banque = self._detecter_banque(txt)
            if banque:
                score += 30
                details["banque"] = banque

        # 2. Analyse RIB (Zone spécifique + Scan global si échec)
        zone_rib = next((f for f in features if f["nom"] == "info_client_rib"), None)
        if zone_rib:
            txt = self._ocr_zone(image_input, zone_rib["zone"])
            full_text_buffer += txt + " "
        
        rib = self._detecter_rib(full_text_buffer)
        
        # Si pas de RIB trouvé, on scanne toute la page (utile pour les formats exotiques)
        if not rib:
            txt_full = pytesseract.image_to_string(image_input, config="--psm 6")
            rib = self._detecter_rib(txt_full)

        if rib:
            score += 40
            details["rib"] = f"{rib[:4]} {rib[4:8]} **** {rib[-2:]}"
            details["rib_raw"] = rib

        # 3. Mots clés financiers
        zone_tab = next((f for f in features if f["nom"] == "tableau_operations"), None)
        if zone_tab:
            txt = self._ocr_zone(image_input, zone_tab["zone"])
            if any(m in txt for m in ["debit", "credit", "solde", "valeur", "date"]):
                score += 20
                details["structure"] = "Tableau financier trouvé"

        est_valide = score >= 50
        return {
            "valide": est_valide,
            "score": score,
            "message": "Relevé valide" if est_valide else "Document inconnu",
            "details": details
        }

# --- FONCTION PORTABLE (MARCHE PARTOUT) ---
def pdf_vers_image_opencv(chemin_pdf):
    """
    Convertit la première page du PDF en image OpenCV sans dépendance système.
    """
    try:
        # Chargement du PDF
        pdf = pdfium.PdfDocument(str(chemin_pdf))
        # Sélection page 1
        page = pdf[0]
        
        # Rendu en haute résolution (scale=3 équivaut à ~200-300 DPI)
        # C'est crucial pour que l'OCR lise bien les petits chiffres du RIB
        bitmap = page.render(scale=3)
        pil_image = bitmap.to_pil()
        
        # Conversion PIL vers OpenCV (RGB -> BGR)
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return opencv_image
        
    except Exception as e:
        print(f"❌ Erreur lecture PDF : {e}")
        return None