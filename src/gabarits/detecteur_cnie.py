import cv2
import numpy as np
import pytesseract
import json
import re
from pathlib import Path

class DetecteurCNIE:
    def __init__(self, chemin_gabarits="models/gabarits/gabarits_maroc.json"):
        self.config = self._charger_gabarits(chemin_gabarits)

    def _charger_gabarits(self, chemin):
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("carte_identite", {})
        except FileNotFoundError:
            return {}

    # --- 1. OCR Amélioré (Zoom + Nettoyage) ---
    def _ocr_global(self, image):
        # 1. Agrandissement (Crucial pour les cartes d'identité)
        h, w = image.shape[:2]
        factor = 1
        if w < 1000: # Si l'image est petite, on zoome
            factor = 2
            image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        
        # 2. Prétraitement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Filtre anti-bruit léger
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Binarisation automatique (Otsu)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # 3. Lecture (PSM 6 = Bloc de texte uniforme, mieux pour les cartes)
        try:
            # On essaie fra+ara, sinon juste fra
            txt = pytesseract.image_to_string(thresh, lang="fra+ara", config="--psm 6")
        except:
            txt = pytesseract.image_to_string(thresh, lang="fra", config="--psm 6")
            
        return txt.lower()

    # --- 2. Couleurs Normalisées ---
    def _analyser_couleurs(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        # Zone Haut (30%) pour la bande rouge
        zone_haut = hsv[0:int(h*0.30), 0:w]
        
        # Zone Globale pour le vert
        zone_globale = hsv

        # Rouge (Début et Fin du spectre HSV)
        # On divise par 255 pour avoir un vrai pourcentage (0.0 à 1.0)
        mask_red1 = cv2.inRange(zone_haut, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(zone_haut, np.array([170, 70, 50]), np.array([180, 255, 255]))
        nb_pixels_rouge = np.sum(mask_red1) + np.sum(mask_red2)
        # total pixels * 255 (car le masque vaut 255)
        ratio_red = nb_pixels_rouge / 255 / (zone_haut.size / 3) 

        # Vert (Teinte 35-85)
        mask_green = cv2.inRange(zone_globale, np.array([35, 40, 40]), np.array([85, 255, 255]))
        nb_pixels_vert = np.sum(mask_green)
        ratio_green = nb_pixels_vert / 255 / (zone_globale.size / 3)

        return {
            # Seuils ajustés : 3% de rouge suffit pour une bande fine
            "rouge_haut": ratio_red > 0.03, 
            # 10% de vert pour l'ancienne carte
            "vert_global": ratio_green > 0.10,
            "valeurs": {"red": round(ratio_red, 3), "green": round(ratio_green, 3)}
        }

    # --- 3. Analyse ---
    def analyser_image(self, chemin_image):
        if isinstance(chemin_image, (str, Path)):
            image = cv2.imread(str(chemin_image))
        else:
            image = chemin_image
            
        if image is None: return {"est_cnie": False, "erreur": "Image invalide"}

        # OCR
        texte_brut = self._ocr_global(image)
        # Couleurs
        couleurs = self._analyser_couleurs(image)

        score_new = 0
        score_old = 0
        details = []

        # -- PISTE 1 : Nouvelle CNIE (Bande Rouge) --
        if couleurs["rouge_haut"]:
            score_new += 30
            details.append("Couleur: Bande Rouge")
        
        # Mots clés spécifiques
        kws_new = ["identite", "nationale", "royaume", "maroc", "valable", "date"]
        matches_new = sum(1 for w in kws_new if w in texte_brut)
        if matches_new >= 2:
            score_new += 40
            details.append(f"Mots-clés ({matches_new})")

        # -- PISTE 2 : Ancienne CNIE (Verte) --
        if couleurs["vert_global"]:
            score_old += 30
            details.append("Couleur: Fond Vert")
            
        kws_old = ["nom", "prenom", "ne le", "a", "fils", "fille", "adresse"]
        matches_old = sum(1 for w in kws_old if w in texte_brut)
        if matches_old >= 2:
            score_old += 40
            details.append(f"Mots-clés ({matches_old})")

        # -- PISTE 3 : CIN (Le Juge de Paix) --
        # Regex robuste : 1 ou 2 lettres + chiffres
        # Ex: AB123456 ou BK 12345
        regex_cin = r"(?<![a-z])[a-z]{1,2}\s?[0-9]{3,6}(?![0-9])"
        match_cin = re.search(regex_cin, texte_brut)
        
        cin_score = 0
        cin_trouve = None
        
        if match_cin:
            cin_trouve = match_cin.group(0).upper()
            cin_score = 50 # Bonus énorme si CIN trouvé
            details.append(f"CIN: {cin_trouve}")
        elif "carte nationale" in texte_brut or "cin" in texte_brut:
             cin_score = 20 # Bonus "Titre trouvé"

        # Score Final
        score_final = max(score_new, score_old) + cin_score
        
        # Détermination du type
        type_doc = "Inconnu"
        if score_final >= 50:
            if score_new > score_old:
                type_doc = "CNIE Biométrique (Nouvelle)"
            else:
                type_doc = "CNIE (Ancienne)"

        # On renvoie aussi un bout du texte lu pour le debug
        snippet = texte_brut[:50].replace('\n', ' ') + "..."

        return {
            "est_cnie": score_final >= 50,
            "type": type_doc,
            "score": min(100, score_final),
            "cin_detecte": cin_trouve,
            "details": details,
            "debug_couleur": couleurs["valeurs"],
            "debug_texte": snippet
        }