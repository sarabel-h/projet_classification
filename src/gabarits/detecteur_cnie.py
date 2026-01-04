"""
Détecteur CNIE marocaine basé sur gabarits JSON
Approche : layout + couleur + mots-clés + CIN
"""

import cv2
import numpy as np
import pytesseract
import json
import re
from pathlib import Path


class DetecteurCNIE:
    def __init__(self, chemin_gabarits="models/gabarits/gabarits_maroc.json"):
        self.gabarits = self._charger_gabarits(chemin_gabarits)
        self.cnie = self.gabarits.get("carte_identite", {})

    # -------------------------
    # Chargement JSON
    # -------------------------
    def _charger_gabarits(self, chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------
    # Utilitaire zone relative → pixels
    # -------------------------
    def _extraire_zone(self, image, zone):
        h, w = image.shape[:2]
        x1 = int(zone[0] * w)
        y1 = int(zone[1] * h)
        x2 = int(zone[2] * w)
        y2 = int(zone[3] * h)
        return image[y1:y2, x1:x2]

    # -------------------------
    # Détection couleur rose/rouge (tolérante)
    # -------------------------
    def _zone_rose_ou_rouge(self, image, seuil=0.15):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bas = np.array([0, 10, 80])
        haut = np.array([180, 160, 255])
        masque = cv2.inRange(hsv, bas, haut)
        ratio = np.sum(masque > 0) / masque.size
        return ratio > seuil

    # -------------------------
    # OCR global (FR + AR)
    # -------------------------
    def _ocr_global(self, image):
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gris = cv2.resize(gris, None, fx=1.5, fy=1.5)
        gris = cv2.GaussianBlur(gris, (3, 3), 0)
        return pytesseract.image_to_string(
            gris,
            lang="fra+ara",
            config="--oem 3 --psm 6"
        )

    # -------------------------
    # Mots-clés CNIE depuis JSON
    # -------------------------
    def _contient_mots_cnie(self, texte):
        structure = self.cnie.get("structure_bande_rouge", {})
        lignes = structure.get("ligne1", {}).get("segments", []) + \
                 structure.get("ligne2", {}).get("segments", [])

        texte = texte.lower()
        for seg in lignes:
            if "texte" in seg:
                if seg["texte"].lower() in texte:
                    return True
        return False

    # -------------------------
    # CIN (regex fiable)
    # -------------------------
    def _detecter_cin(self, texte):
        return re.search(r"\b[A-Z]{1,2}\d{5,6}\b", texte) is not None

    # -------------------------
    # Vérification format carte
    # -------------------------
    def _format_carte_ok(self, image):
        h, w = image.shape[:2]
        ratio = w / h
        attendu = self.cnie.get("features", [])[-1].get("valeur", 1.586)
        return abs(ratio - attendu) < 0.15, ratio

    # -------------------------
    # Analyse principale
    # -------------------------
    def analyser_image(self, chemin_image):
        image = cv2.imread(str(chemin_image))
        if image is None:
            return {"est_cnie": False, "erreur": "Image non chargée"}

        score = 0

        # 1. Format carte
        format_ok, ratio = self._format_carte_ok(image)
        if format_ok:
            score += 20

        # 2. Bande haute couleur
        for f in self.cnie.get("features", []):
            if f["nom"] == "bande_rouge_haut":
                zone = self._extraire_zone(image, f["zone"])
                if self._zone_rose_ou_rouge(zone):
                    score += 20

        # 3. OCR + mots-clés
        texte = self._ocr_global(image)
        if self._contient_mots_cnie(texte):
            score += 30

        # 4. CIN
        if self._detecter_cin(texte):
            score += 30

        return {
            "est_cnie": score >= 60,
            "score": score,
            "ratio": round(ratio, 3),
            "message": "CNIE détectée" if score >= 60 else "Pas une CNIE"
        }


# -------------------------
# TEST
# -------------------------
if __name__ == "__main__":
    detecteur = DetecteurDocuments(
        chemin_gabarits="models/gabarits/gabarits_maroc.json"
    )

    image_test = "tests/image_test.jpg"
    if Path(image_test).exists():
        resultat = detecteur.analyser_image(image_test)
        print(resultat)
    else:
        print("Image de test introuvable")