"""
Script pour télécharger automatiquement les modèles pré-entraînés
"""

import torch
import torchvision.models as models
from transformers import CamembertModel, CamembertTokenizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_cv_models():
    """Télécharge les modèles Computer Vision"""
    models_cv_dir = Path("models/cv")
    models_cv_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Téléchargement de ResNet50...")
    resnet50 = models.resnet50(pretrained=True)
    torch.save(resnet50.state_dict(), models_cv_dir / "resnet50_pretrained.pth")
    logger.info("✓ ResNet50 téléchargé et sauvegardé")
    
    logger.info("Téléchargement d'EfficientNet...")
    efficientnet = models.efficientnet_b0(pretrained=True)
    torch.save(efficientnet.state_dict(), models_cv_dir / "efficientnet_b0_pretrained.pth")
    logger.info("✓ EfficientNet téléchargé et sauvegardé")


def download_nlp_models():
    """Télécharge les modèles NLP"""
    models_nlp_dir = Path("models/nlp")
    models_nlp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Téléchargement de CamemBERT...")
    model_name = "camembert-base"
    
    # Télécharger le modèle
    model = CamembertModel.from_pretrained(model_name)
    model.save_pretrained(models_nlp_dir / "camembert-base")
    
    # Télécharger le tokenizer
    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(models_nlp_dir / "camembert-base")
    
    logger.info("✓ CamemBERT téléchargé et sauvegardé")


def main():
    """Télécharge tous les modèles"""
    logger.info("="*60)
    logger.info("TÉLÉCHARGEMENT DES MODÈLES PRÉ-ENTRAÎNÉS")
    logger.info("="*60)
    
    try:
        download_cv_models()
        download_nlp_models()
        
        logger.info("\n" + "="*60)
        logger.info("✓ TOUS LES MODÈLES TÉLÉCHARGÉS AVEC SUCCÈS")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {e}")
        logger.error("Vérifiez votre connexion internet")


if __name__ == "__main__":
    main()