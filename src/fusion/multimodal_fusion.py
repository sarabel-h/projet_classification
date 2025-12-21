"""
Module de fusion multimodale pour combiner les prédictions CV et NLP
"""

import logging
from typing import Dict, Tuple, Optional, List
import numpy as np

logger = logging.getLogger(__name__)


class MultimodalFusion:
    """Fusionne les prédictions de CV et NLP pour une décision robuste"""
    
    def __init__(self):
        """Initialise le système de fusion"""
        self.confidence_threshold = 0.5
        self.high_confidence_threshold = 0.8
        self.very_high_confidence_threshold = 0.9
        
        # Règles métier par classe
        self.business_rules = {
            "piece_identite": self._validate_piece_identite,
            "releve_bancaire": self._validate_releve_bancaire,
            "facture_electricite": self._validate_facture_electricite,
            "facture_eau": self._validate_facture_eau,
            "document_employeur": self._validate_document_employeur
        }
    
    def fuse_predictions(self,
                        cv_prediction: Dict[str, float],
                        nlp_prediction: Dict[str, float],
                        gabarit_scores: Dict[str, float],
                        semantic_patterns: Dict[str, float]) -> Tuple[str, float, str]:
        """
        Fusionne les prédictions de plusieurs modèles
        
        Args:
            cv_prediction: Prédictions du modèle CV {classe: confiance}
            nlp_prediction: Prédictions du modèle NLP {classe: confiance}
            gabarit_scores: Scores des gabarits {classe: score}
            semantic_patterns: Scores des motifs sémantiques {classe: score}
            
        Returns:
            Tuple (classe_finale, confiance, raison_decision)
        """
        
        # Obtenir les meilleurs prédictions
        cv_class = max(cv_prediction, key=cv_prediction.get)
        cv_conf = cv_prediction[cv_class]
        
        nlp_class = max(nlp_prediction, key=nlp_prediction.get)
        nlp_conf = nlp_prediction[nlp_class]
        
        # Cas 1: Accord parfait entre CV et NLP avec haute confiance
        if cv_class == nlp_class and cv_conf > self.high_confidence_threshold and nlp_conf > self.high_confidence_threshold:
            final_conf = (cv_conf + nlp_conf) / 2
            reason = "Accord parfait CV+NLP"
            logger.info(f"[FUSION] Accord parfait: {cv_class} (confiance: {final_conf:.2f})")
            return cv_class, final_conf, reason
        
        # Cas 2: CV très confiant + validation par gabarits
        if cv_conf > self.very_high_confidence_threshold:
            gabarit_score = gabarit_scores.get(cv_class, 0.5)
            if gabarit_score > 0.7:
                final_conf = 0.9 * cv_conf + 0.1 * gabarit_score
                reason = "CV fort + validation gabarits"
                logger.info(f"[FUSION] CV confiant: {cv_class} (confiance: {final_conf:.2f})")
                return cv_class, final_conf, reason
        
        # Cas 3: NLP très confiant + motifs textuels spécifiques
        if nlp_conf > self.very_high_confidence_threshold:
            semantic_score = semantic_patterns.get(nlp_class, 0.5)
            if semantic_score > 0.7:
                final_conf = 0.9 * nlp_conf + 0.1 * semantic_score
                reason = "NLP fort + motifs textuels"
                logger.info(f"[FUSION] NLP confiant: {nlp_class} (confiance: {final_conf:.2f})")
                return nlp_class, final_conf, reason
        
        # Cas 4: Fusion pondérée si confiances modérées
        if cv_conf > self.confidence_threshold and nlp_conf > self.confidence_threshold:
            # Calculer les scores moyens pondérés pour toutes les classes
            classes = set(cv_prediction.keys())
            final_scores = {}
            
            for class_name in classes:
                cv_score = cv_prediction.get(class_name, 0.0)
                nlp_score = nlp_prediction.get(class_name, 0.0)
                gabarit_score = gabarit_scores.get(class_name, 0.5)
                semantic_score = semantic_patterns.get(class_name, 0.5)
                
                # Fusion pondérée
                final_scores[class_name] = (0.4 * cv_score + 
                                           0.35 * nlp_score + 
                                           0.15 * gabarit_score + 
                                           0.1 * semantic_score)
            
            final_class = max(final_scores, key=final_scores.get)
            final_conf = final_scores[final_class]
            reason = "Fusion pondérée CV+NLP+gabarits"
            logger.info(f"[FUSION] Fusion pondérée: {final_class} (confiance: {final_conf:.2f})")
            return final_class, final_conf, reason
        
        # Cas 5: Confiance faible - favoriser le gabarit
        best_gabarit = max(gabarit_scores, key=gabarit_scores.get)
        if gabarit_scores[best_gabarit] > 0.6:
            final_conf = gabarit_scores[best_gabarit]
            reason = "Fallback sur gabarits"
            logger.info(f"[FUSION] Fallback gabarits: {best_gabarit} (confiance: {final_conf:.2f})")
            return best_gabarit, final_conf, reason
        
        # Cas 6: Confiance très faible - rejeter
        reason = "Confiance insuffisante (à vérifier manuellement)"
        logger.warning(f"[FUSION] Document rejeté: confiance insuffisante")
        return "à_verifier", 0.3, reason
    
    def _validate_piece_identite(self, features: Dict) -> bool:
        """Valide une carte d'identité"""
        # Doit avoir une photo
        has_photo = features.get("presence_photo", False)
        # Doit avoir le bon ratio d'aspect
        correct_ratio = features.get("correct_aspect_ratio", False)
        return has_photo and correct_ratio
    
    def _validate_releve_bancaire(self, features: Dict) -> bool:
        """Valide un relevé bancaire"""
        # Doit avoir une structure tabulaire
        has_structure = features.get("structure_tabulaire", False)
        # Doit contenir des montants financiers
        has_amounts = features.get("contains_amounts", False)
        return has_structure and has_amounts
    
    def _validate_facture_electricite(self, features: Dict) -> bool:
        """Valide une facture d'électricité"""
        # Doit avoir une structure tabulaire
        has_structure = features.get("structure_tabulaire", False)
        # Doit contenir des mots clés (kWh, électricité, etc.)
        has_keywords = features.get("has_electricity_keywords", False)
        return has_structure and has_keywords
    
    def _validate_facture_eau(self, features: Dict) -> bool:
        """Valide une facture d'eau"""
        # Doit avoir une structure tabulaire
        has_structure = features.get("structure_tabulaire", False)
        # Doit contenir des mots clés (m³, eau, etc.)
        has_keywords = features.get("has_water_keywords", False)
        return has_structure and has_keywords
    
    def _validate_document_employeur(self, features: Dict) -> bool:
        """Valide un document employeur"""
        # Doit avoir du texte dense
        has_text = features.get("densite_texte", 0) > 0.1
        # Doit contenir des mentions salariales
        has_salary_keywords = features.get("has_salary_keywords", False)
        return has_text or has_salary_keywords
    
    def apply_business_rules(self, predicted_class: str, features: Dict, confidence: float) -> Tuple[str, float, str]:
        """
        Applique les règles métier pour valider une prédiction
        
        Args:
            predicted_class: Classe prédite
            features: Features détectées
            confidence: Confiance de la prédiction
            
        Returns:
            Tuple (classe_validée, confiance_finale, raison)
        """
        if predicted_class not in self.business_rules:
            return predicted_class, confidence, "Pas de règle métier"
        
        validator = self.business_rules[predicted_class]
        is_valid = validator(features)
        
        if not is_valid:
            # Si la règle n'est pas satisfaite, réduire la confiance
            confidence *= 0.7
            reason = f"Règle métier non satisfaite pour {predicted_class}"
            
            if confidence < 0.5:
                return "à_verifier", confidence, reason
        
        return predicted_class, confidence, "Validation réussie"
    
    def create_confidence_report(self,
                                predicted_class: str,
                                confidence: float,
                                cv_scores: Dict,
                                nlp_scores: Dict,
                                gabarit_scores: Dict,
                                semantic_scores: Dict) -> Dict:
        """
        Crée un rapport détaillé de confiance
        
        Args:
            predicted_class: Classe prédite
            confidence: Confiance finale
            cv_scores: Scores CV
            nlp_scores: Scores NLP
            gabarit_scores: Scores gabarits
            semantic_scores: Scores sémantiques
            
        Returns:
            Dict avec tous les détails
        """
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "cv_scores": cv_scores,
            "nlp_scores": nlp_scores,
            "gabarit_scores": gabarit_scores,
            "semantic_scores": semantic_scores,
            "requires_manual_review": confidence < 0.6 or predicted_class == "à_verifier"
        }
    
    def batch_fuse_predictions(self,
                               cv_predictions: List[Dict],
                               nlp_predictions: List[Dict],
                               gabarit_scores_list: List[Dict],
                               semantic_patterns_list: List[Dict]) -> List[Tuple[str, float, str]]:
        """
        Fusionne les prédictions pour un lot de documents
        
        Args:
            cv_predictions: Liste des prédictions CV
            nlp_predictions: Liste des prédictions NLP
            gabarit_scores_list: Liste des scores gabarits
            semantic_patterns_list: Liste des motifs sémantiques
            
        Returns:
            Liste des résultats fusionnés
        """
        results = []
        
        for cv_pred, nlp_pred, gab_scores, sem_patterns in zip(
            cv_predictions, nlp_predictions, gabarit_scores_list, semantic_patterns_list
        ):
            result = self.fuse_predictions(cv_pred, nlp_pred, gab_scores, sem_patterns)
            results.append(result)
        
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fusion = MultimodalFusion()
    logger.info("MultimodalFusion initialized")
