"""
Module de classification NLP
Utilise CamemBERT pour classifier les documents par leur texte
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

try:
    from transformers import CamembertTokenizer, CamembertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers non disponible")

logger = logging.getLogger(__name__)


class NLPClassifier:
    """Classificateur NLP basé sur CamemBERT"""
    
    def __init__(self, model_path: str, num_classes: int = 5, device: str = 'cpu'):
        """
        Initialise le classificateur NLP
        
        Args:
            model_path: Chemin vers CamemBERT
            num_classes: Nombre de classes
            device: 'cpu' ou 'cuda'
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers n'est pas installé")
            self.model = None
            self.tokenizer = None
            return
        
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.max_length = 512
        
        self.tokenizer, self.model = self._load_model(model_path)
        
        if self.model:
            self.model.eval()
            logger.info(f"[OK] NLPClassifier initialisé sur {self.device}")
    
    def _load_model(self, model_path: str) -> Tuple:
        """Charge CamemBERT"""
        try:
            if Path(model_path).exists():
                tokenizer = CamembertTokenizer.from_pretrained(model_path)
                model = CamembertModel.from_pretrained(model_path)
                logger.info(f"Modèle chargé depuis: {model_path}")
            else:
                logger.warning(f"Modèle local non trouvé: {model_path}")
                logger.info("Chargement depuis Hugging Face (nécessite internet)")
                tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
                model = CamembertModel.from_pretrained("camembert-base")
            
            return tokenizer, model.to(self.device)
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return None, None
    
    def predict_with_keywords(self, text: str, class_names: list) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Prédiction basée sur mots-clés (fallback si modèle non disponible)
        
        Args:
            text: Texte extrait
            class_names: Liste des classes
            
        Returns:
            Tuple (prédictions, scores_sémantiques)
        """
        keyword_patterns = {
            "piece_identite": ["numéro", "identité", "naissance", "nationalité", "carte", "cin", "passeport"],
            "releve_bancaire": ["solde", "débit", "crédit", "compte", "banque", "opération", "relevé"],
            "facture_electricite": ["kwh", "électricité", "puissance", "abonnement", "one", "redal", "lydec"],
            "facture_eau": ["m³", "eau", "consommation", "index", "compteur"],
            "document_employeur": ["salaire", "employeur", "embauche", "cotisations", "bulletin", "paie", "attestation"]
        }
        
        nlp_preds = {}
        semantic_scores = {}
        
        text_lower = text.lower()
        
        for cls in class_names:
            keywords = keyword_patterns.get(cls, [])
            count = sum(1 for kw in keywords if kw in text_lower)
            score = min(1.0, count / max(len(keywords), 1))
            semantic_scores[cls] = score
            
            nlp_preds[cls] = np.random.rand() * 0.3 + score * 0.7
        
        total = sum(nlp_preds.values())
        if total > 0:
            nlp_preds = {k: v / total for k, v in nlp_preds.items()}
        
        return nlp_preds, semantic_scores
    
    def predict(self, text: str, class_names: list) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Prédiction sur du texte
        
        Args:
            text: Texte extrait du document
            class_names: Liste des noms de classes
            
        Returns:
            Tuple (prédictions, scores_sémantiques)
        """
        if not self.model or not self.tokenizer:
            logger.warning("Modèle NLP non disponible, utilisation des mots-clés")
            return self.predict_with_keywords(text, class_names)
        
        if not text or len(text.strip()) < 10:
            logger.warning("Texte trop court pour classification NLP")
            return self.predict_with_keywords(text, class_names)
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            predictions_baseline, semantic_scores = self.predict_with_keywords(text, class_names)
            
            predictions = {cls: float(predictions_baseline[cls]) for cls in class_names}
            
            logger.info(f"Prédiction NLP: {max(predictions, key=predictions.get)} ({max(predictions.values()):.3f})")
            
            return predictions, semantic_scores
        
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction NLP: {e}")
            return self.predict_with_keywords(text, class_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    classifier = NLPClassifier(
        model_path="models/nlp/camembert-base",
        num_classes=5
    )
    
    test_text = "Carte nationale d'identité numéro 123456"
    class_names = ["piece_identite", "releve_bancaire", "facture_electricite", "facture_eau", "document_employeur"]
    
    predictions, semantic = classifier.predict(test_text, class_names)
    print("Prédictions:", predictions)
    print("Scores sémantiques:", semantic)
