"""
Module de gabarits pour la détection de structures de documents
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GabaritDetector:
    """Détecte les caractéristiques structurelles des documents basées sur les gabarits"""
    
    def __init__(self, gabarits_config_path: str = "models/gabarits/gabarits.json"):
        """
        Initialise le détecteur de gabarits
        
        Args:
            gabarits_config_path: Chemin vers la configuration des gabarits
        """
        self.gabarits_config = self._load_gabarits(gabarits_config_path)
        self.feature_weights = {
            "presence_photo": 0.25,
            "ratio_aspect": 0.20,
            "densite_texte": 0.20,
            "structure_tabulaire": 0.20,
            "zone_signature": 0.15
        }
    
    def _load_gabarits(self, config_path: str) -> Dict:
        """Charge la configuration des gabarits"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Configuration par défaut
        return {
            "piece_identite": {
                "zones_obligatoires": ["photo", "texte_identité", "numéro", "dates"],
                "caracteristiques": ["ratio_carte", "presence_photo", "elements_securite"],
                "ratio_aspect_attendu": 1.58,
                "tolerance_ratio": 0.1
            },
            "releve_bancaire": {
                "caracteristiques": ["alignement_vertical", "separateurs_visibles", "montants"],
                "structure": "tabulaire",
                "nb_colonnes_min": 3
            },
            "facture_electricite": {
                "caracteristiques": ["tableau_consommation", "unites_kwh", "periodes"],
                "mots_cles": ["kWh", "électricité", "puissance", "abonnement"],
                "structure": "tabulaire"
            },
            "facture_eau": {
                "caracteristiques": ["tableau_consommation", "unites_m3"],
                "mots_cles": ["m³", "eau", "consommation", "index"],
                "structure": "tabulaire"
            },
            "document_employeur": {
                "caracteristiques": ["en_tete_entreprise", "zone_signature", "mentions_salariales"],
                "mots_cles": ["salaire", "employeur", "cotisations", "bulletin"],
                "structure": "libre"
            }
        }
    
    def detect_photo_region(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte la présence d'une région de photo
        
        Args:
            image: Image numpy
            
        Returns:
            Tuple (présence_photo, confiance)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Détecter les visages (approximation pour les photos)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        has_photo = len(faces) > 0
        confidence = 0.8 if has_photo else 0.2
        
        return has_photo, confidence
    
    def detect_aspect_ratio(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Détecte le ratio d'aspect (carte vs A4)
        
        Args:
            image: Image numpy
            
        Returns:
            Tuple (ratio, type_document)
        """
        height, width = image.shape[:2]
        ratio = width / height
        
        # Identité: ~1.58 (85.6mm x 53.98mm)
        # A4: ~1.41 (210mm x 297mm)
        # A5: ~1.41
        
        if 1.4 < ratio < 1.65:
            if abs(ratio - 1.58) < 0.1:
                doc_type = "carte_identite"
            else:
                doc_type = "format_a4"
        else:
            doc_type = "format_autre"
        
        return ratio, doc_type
    
    def detect_text_density(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Détecte la densité de texte dans l'image
        
        Args:
            image: Image numpy
            
        Returns:
            Tuple (densité, type_contenu)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Détecter les caractères avec Canny
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculer la densité
        total_pixels = edges.size
        text_pixels = np.count_nonzero(edges)
        density = text_pixels / total_pixels
        
        if density < 0.05:
            content_type = "peu_de_texte"
        elif density < 0.15:
            content_type = "texte_modere"
        else:
            content_type = "beaucoup_de_texte"
        
        return density, content_type
    
    def detect_tabular_structure(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte la présence d'une structure tabulaire (lignes et colonnes)
        
        Args:
            image: Image numpy
            
        Returns:
            Tuple (présence_structure, confiance)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Binariser
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Détecter les lignes avec la transformée de Hough
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return False, 0.1
        
        horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 5)
        vertical_lines = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 5)
        
        has_structure = horizontal_lines > 2 and vertical_lines > 2
        confidence = min(0.9, (horizontal_lines + vertical_lines) / 20.0)
        
        return has_structure, confidence
    
    def detect_signature_zone(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Détecte la présence d'une zone de signature
        
        Args:
            image: Image numpy
            
        Returns:
            Tuple (présence_signature, confiance)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Les signatures ont généralement une texture particulière
        # Analyser le bas du document
        height = gray.shape[0]
        bottom_section = gray[int(height*0.7):height]
        
        # Détcter les traits (signature typique)
        edges = cv2.Canny(bottom_section, 50, 150)
        
        has_signature = np.count_nonzero(edges) > (edges.size * 0.05)
        confidence = 0.7 if has_signature else 0.2
        
        return has_signature, confidence
    
    def compute_gabarit_score(self, image: np.ndarray, document_class: str) -> Dict[str, float]:
        """
        Calcule les scores de correspondance avec tous les gabarits
        
        Args:
            image: Image numpy
            document_class: Classe de document attendue
            
        Returns:
            Dict des scores par gabarit
        """
        scores = {}
        
        # Extraire les features
        has_photo, photo_conf = self.detect_photo_region(image)
        ratio, ratio_type = self.detect_aspect_ratio(image)
        density, density_type = self.detect_text_density(image)
        has_structure, structure_conf = self.detect_tabular_structure(image)
        has_signature, signature_conf = self.detect_signature_zone(image)
        
        features = {
            "presence_photo": photo_conf,
            "ratio_aspect": 1.0 - abs(ratio - 1.58) if ratio_type == "carte_identite" else 0.3,
            "densite_texte": density,
            "structure_tabulaire": structure_conf,
            "zone_signature": signature_conf
        }
        
        # Calculer les scores pondérés pour chaque classe
        for class_name, gabarit in self.gabarits_config.items():
            score = 0.0
            
            # Implémenter la logique de scoring pour chaque classe
            if class_name == "piece_identite":
                score = (features["presence_photo"] * 0.3 + 
                        features["ratio_aspect"] * 0.4 +
                        (1.0 - features["densite_texte"]) * 0.3)
            
            elif class_name == "releve_bancaire":
                score = (features["structure_tabulaire"] * 0.5 +
                        features["densite_texte"] * 0.3 +
                        (1.0 - features["presence_photo"]) * 0.2)
            
            elif class_name in ["facture_electricite", "facture_eau"]:
                score = (features["structure_tabulaire"] * 0.4 +
                        features["densite_texte"] * 0.4 +
                        (1.0 - features["presence_photo"]) * 0.2)
            
            elif class_name == "document_employeur":
                score = (features["zone_signature"] * 0.3 +
                        features["densite_texte"] * 0.4 +
                        (1.0 - features["presence_photo"]) * 0.3)
            
            scores[class_name] = min(1.0, max(0.0, score))
        
        return scores
    
    def validate_predictions(self, predicted_class: str, features_dict: Dict) -> bool:
        """
        Valide une prédiction selon les règles métier
        
        Args:
            predicted_class: Classe prédite
            features_dict: Dict des features détectées
            
        Returns:
            True si la prédiction est valide
        """
        gabarit = self.gabarits_config.get(predicted_class)
        
        if predicted_class == "piece_identite":
            return features_dict.get("presence_photo", False)
        
        elif predicted_class == "releve_bancaire":
            return features_dict.get("structure_tabulaire", False)
        
        elif predicted_class in ["facture_electricite", "facture_eau"]:
            return features_dict.get("structure_tabulaire", False)
        
        elif predicted_class == "document_employeur":
            return features_dict.get("densite_texte", 0) > 0.1
        
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = GabaritDetector()
    print("Gabarits chargés:")
    for class_name in detector.gabarits_config.keys():
        print(f"  - {class_name}")
