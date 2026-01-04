import cv2
import numpy as np
import pytesseract
import json
import re
import unicodedata
import pypdfium2 as pdfium
from pathlib import Path

class DetecteurEmployeur:
    def __init__(self, chemin_gabarits="gabarits_maroc.json"):
        self.config = self._charger_gabarits(chemin_gabarits)

    def _charger_gabarits(self, chemin):
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("attestation_employeur", {})
        except FileNotFoundError:
            print(f"⚠️ Erreur : {chemin} introuvable.")
            return {}

    def _nettoyer_texte(self, texte):
        """Enlève les accents et met en minuscule pour faciliter la recherche"""
        nfkd_form = unicodedata.normalize('NFKD', texte)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()

    def _ocr_zone(self, image):
        """OCR optimisé pour le document entier"""
        # Conversion niveau de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Légère augmentation du contraste
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        # Binarisation OTSU
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return pytesseract.image_to_string(thresh, lang="fra", config="--psm 6")

    def _detecter_cnss(self, texte_brut):
        """Cherche CNSS avec regex large"""
        # On cherche dans le texte brut pour garder les chiffres exacts
        match = re.search(r"(cnss|immatriculé|n°|affilie)\D{0,15}(\d{9})", texte_brut.lower())
        if match:
            return match.group(2)
        return None

    def _detecter_secteur(self, texte_clean):
        """Détection améliorée secteur public/privé"""
        texte_lower = texte_clean.lower()
        
        # Mots forts pour secteur PUBLIC
        mots_public_forts = [
            "royaume du maroc", "ministère", "ministre", 
            "wilaya", "préfecture", "commune", "municipalité",
            "éducation nationale", "santé publique", "affaires étrangères",
            "administration", "fonction publique", "état", "gouvernement"
        ]
        
        # Mots pour secteur PRIVÉ
        mots_prive = [
            "s.a.", "s.a.r.l.", "sarl", "eurl", "snc",
            "entreprise", "société", "compagnie", "holding",
            "limitée", "capital social", "commerciale", "bim stores"
        ]
        
        # Compter les occurrences
        score_public = 0
        score_prive = 0
        
        for mot in mots_public_forts:
            if mot in texte_lower:
                score_public += 2  # Plus de poids pour les mots forts
        
        # Vérifier présence de "Royaume" OU "Maroc" séparément (tolérance OCR)
        if "royaume" in texte_lower or "hoyaume" in texte_lower:
            score_public += 1
        if "maroc" in texte_lower or "mar0c" in texte_lower:  # tolérance OCR
            score_public += 1
        
        for mot in mots_prive:
            if mot in texte_lower:
                score_prive += 2  # Augmenté le poids
        
        # Vérifier les formats de numéro CNSS
        # CNSS standard privé: 9 chiffres
        cnss_patterns = re.findall(r'\b\d{9}\b', texte_lower)
        if cnss_patterns:
            # Si CNSS numérique simple, plus probablement privé
            score_prive += 1
        
        # Décision avec seuil
        if score_public > score_prive and score_public >= 2:
            return "Public", score_public
        elif score_prive > score_public and score_prive >= 2:
            return "Privé", score_prive
        else:
            return "Indéterminé", 0

    def _detecter_salaire_amelioré(self, texte_brut):
        """Détection intelligente du salaire améliorée"""
        lignes = texte_brut.lower().split('\n')
        
        # Regex améliorée pour montant
        regex_montant = r"(\d{1,3}(?:[\s\.]\d{3})*(?:[.,]\d{2})?)\s*(?:dh|mad|dirham|dhs)?"
        
        meilleur_candidat = None
        meilleur_score = -1
        
        for i, ligne in enumerate(lignes):
            # Ignorer les lignes non pertinentes
            if any(mot in ligne for mot in ["capital", "s.a", "sarl", "au capital"]):
                continue
                
            # Rechercher montant
            match = re.search(regex_montant, ligne)
            if match:
                montant_str = match.group(1).replace(" ", "").replace(".", "")
                
                # Convertir en float
                try:
                    montant_num = float(montant_str.replace(",", "."))
                except:
                    continue
                
                # Filtre de plage réaliste pour salaire marocain
                if not (1000 <= montant_num <= 50000):
                    continue
                
                score_ligne = 0
                
                # Mots-clés forts pour salaire
                mots_forts = ["net à payer", "salaire net", "émoluments nets", "perçoit", "perçoit un"]
                mots_moyens = ["net", "brut", "salaire", "mensuel", "rémunération", "traitement"]
                mots_faibles = ["montant", "somme", "total", "dh", "mad"]
                
                # Calculer le score
                for mot in mots_forts:
                    if mot in ligne:
                        score_ligne += 3
                
                for mot in mots_moyens:
                    if mot in ligne:
                        score_ligne += 2
                
                for mot in mots_faibles:
                    if mot in ligne:
                        score_ligne += 1
                
                # Bonus si la ligne est dans les premières lignes (plus probable d'être le salaire)
                if i < 10:
                    score_ligne += 1
                
                # Garder le meilleur candidat
                if score_ligne > meilleur_score:
                    meilleur_score = score_ligne
                    meilleur_candidat = f"{montant_num:,.2f} DH".replace(",", " ").replace(".", ",")
        
        return meilleur_candidat if meilleur_score >= 2 else None

    def analyser(self, image_input):
        if image_input is None:
            return {"valide": False, "erreur": "Image vide"}

        score = 0
        details = {}
        
        # OCR Complet
        texte_brut = self._ocr_zone(image_input)
        texte_clean = self._nettoyer_texte(texte_brut)

        # 1. Identification Type
        type_doc = "Inconnu"
        
        is_salaire = any(k in texte_clean for k in ["salaire", "émoluments", "traitement", "rémunération"])
        is_travail = any(k in texte_clean for k in ["travail", "employé", "fonction", "emploi", "poste"])
        is_certif = any(k in texte_clean for k in ["certificat", "attestation", "certification"])

        if is_salaire and is_certif:
            type_doc = "Attestation de Salaire"
            score += 40
        elif is_travail and is_certif:
            type_doc = "Attestation de Travail"
            score += 35
        elif is_salaire:
            type_doc = "Document Salarial"
            score += 30
        elif is_travail:
            type_doc = "Document de Travail"
            score += 25

        # 2. Détection secteur améliorée
        secteur, score_secteur = self._detecter_secteur(texte_clean)
        if secteur != "Indéterminé":
            details["secteur"] = secteur
            score += 15
            if secteur == "Public":
                score += 5  # Bonus pour secteur public

        # 3. Détection CNSS
        cnss = self._detecter_cnss(texte_brut)
        if cnss:
            score += 20
            details["cnss"] = cnss
            # Si CNSS trouvé et secteur pas encore défini, c'est probablement privé
            if "secteur" not in details:
                details["secteur"] = "Privé"
                score += 5

        # 4. Détection Salaire améliorée
        salaire = self._detecter_salaire_amelioré(texte_brut)
        if salaire:
            score += 15
            details["salaire_detecte"] = salaire
            # Si salaire trouvé, renforcer le type
            if "salaire" not in type_doc.lower():
                type_doc = "Attestation de Salaire"
                score += 5

        # 5. Extraction Employeur
        match_employeur = re.search(r"(nous|je)\s+soussign[ée]s?,?\s+([a-zA-Z0-9\s\.\-]+)(?=,|$)", texte_clean)
        if not match_employeur:
            # Autre pattern: "La société X"
            match_employeur = re.search(r"la\s+(?:société|entreprise|compagnie)\s+([a-zA-Z0-9\s\.\-]+)", texte_clean)
        
        if match_employeur:
            soc = match_employeur.group(2 if match_employeur.lastindex == 2 else 1).strip().upper()
            if len(soc) > 3 and soc not in ["SIGNE", "SOUSSIGNE", "SOUSSIGNES"]:
                details["employeur"] = soc
                score += 10

        # 6. Vérifier présence de signature
        if "signature" in texte_clean or "signé" in texte_clean or "cachet" in texte_clean:
            details["signature"] = True
            score += 5

        # 7. Vérifier date
        date_match = re.search(r"\d{1,2}/\d{1,2}/\d{4}", texte_brut)
        if date_match:
            details["date"] = date_match.group()
            score += 5

        # Seuil de validation
        est_valide = score >= 50
        
        return {
            "valide": est_valide,
            "type_document": type_doc,
            "score": score,
            "details": details,
            "texte_analyse": texte_brut[:500] + "..." if len(texte_brut) > 500 else texte_brut
        }

# --- FONCTION PDF ---
def pdf_vers_image_opencv(chemin_pdf):
    try:
        pdf = pdfium.PdfDocument(str(chemin_pdf))
        page = pdf[0]
        bitmap = page.render(scale=3) 
        pil_image = bitmap.to_pil()
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"❌ Erreur PDF : {e}")
        return None