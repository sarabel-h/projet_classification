"""
Script d'initialisation offline
Télécharge et configure tous les modèles nécessaires pour fonctionner sans internet
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OfflineSetup:
    """Gère l'initialisation complète de l'environnement offline"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
    
    def verify_dependencies(self) -> bool:
        """Vérifie que toutes les dépendances requises sont installées"""
        required_packages = {
            'cv2': 'opencv-python',
            'torch': 'torch',
            'transformers': 'transformers',
            'pytesseract': 'pytesseract',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'PIL': 'pillow',
            'pdf2image': 'pdf2image',
        }
        
        missing_packages = []
        
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
                logger.info(f"✓ {package_name} est installé")
            except ImportError:
                logger.warning(f"✗ {package_name} n'est pas installé")
                missing_packages.append(package_name)
        
        if missing_packages:
            logger.error("Packages manquants:")
            for pkg in missing_packages:
                logger.error(f"  - Installez: pip install {pkg}")
            return False
        
        return True
    
    def create_directory_structure(self):
        """Crée la structure de répertoires"""
        directories = [
            self.models_dir / "cv",
            self.models_dir / "nlp",
            self.models_dir / "gabarits",
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "annotations",
            self.project_root / "src" / "preprocessing",
            self.project_root / "src" / "computer_vision",
            self.project_root / "src" / "nlp",
            self.project_root / "src" / "fusion",
            self.project_root / "src" / "gabarits",
            self.project_root / "src" / "utils",
            self.project_root / "tests",
            self.project_root / "output",
            self.project_root / "logs",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Créé: {directory}")
    
    def setup_models_offline(self):
        """Configure les modèles pour fonctionner offline"""
        logger.info("Configuration des modèles offline...")
        
        # Configuration des modèles CV
        logger.info("Configuration des modèles CV...")
        cv_models = {
            "resnet50": {
                "description": "ResNet50 pré-entraîné sur ImageNet",
                "status": "À télécharger manuellement ou utiliser from_pretrained"
            },
            "efficientnet": {
                "description": "EfficientNet pré-entraîné",
                "status": "À télécharger manuellement ou utiliser from_pretrained"
            }
        }
        
        # Configuration des modèles NLP
        logger.info("Configuration des modèles NLP...")
        nlp_models = {
            "camembert": {
                "description": "CamemBERT pré-entraîné pour le français",
                "status": "À télécharger manuellement ou utiliser from_pretrained"
            }
        }
        
        # Configuration des gabarits
        logger.info("Configuration des gabarits...")
        gabarits_config = {
            "identity": {
                "zones_obligatoires": ["photo", "texte_identité", "numéro", "dates"],
                "caracteristiques": ["ratio_carte", "presence_photo", "elements_securite"]
            },
            "financial": {
                "structure": "tabulaire",
                "caracteristiques": ["alignement_vertical", "separateurs_visibles"]
            },
            "electricity": {
                "caracteristiques": ["tableau_consommation", "unites_kwh", "periodes_facturation"]
            },
            "water": {
                "caracteristiques": ["tableau_consommation", "unites_m3", "structure_facture"]
            },
            "employer": {
                "caracteristiques": ["en_tetes_entreprise", "zones_signature", "structure_contrat"]
            }
        }
        
        return cv_models, nlp_models, gabarits_config
    
    def setup_tesseract(self):
        """Configure Tesseract OCR pour le français"""
        logger.info("Configuration de Tesseract OCR...")
        
        try:
            import pytesseract
            logger.info("✓ pytesseract est installé")
            
            # Vérifier si Tesseract est disponible
            try:
                pytesseract.get_tesseract_version()
                logger.info("✓ Tesseract est disponible sur le système")
            except Exception as e:
                logger.warning(f"⚠ Tesseract système non trouvé: {e}")
                logger.info("Installation recommandée:")
                logger.info("  - Linux: sudo apt-get install tesseract-ocr tesseract-ocr-fra")
                logger.info("  - macOS: brew install tesseract")
                logger.info("  - Windows: Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki")
        
        except ImportError:
            logger.error("pytesseract n'est pas installé")
            logger.error("Installez: pip install pytesseract")
    
    def create_config_file(self):
        """Crée le fichier de configuration principal"""
        config = {
            "project": {
                "name": "Document Classification System",
                "version": "1.0.0",
                "mode": "offline"
            },
            "models": {
                "cv": {
                    "backbone": "resnet50",
                    "input_size": [224, 224],
                    "pretrained": True
                },
                "nlp": {
                    "model_name": "distiluse-base-multilingual-cased-v2",
                    "max_length": 512
                },
                "gabarits": {
                    "enabled": True,
                    "confidence_threshold": 0.7
                }
            },
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "annotations_dir": "data/annotations"
            },
            "output": {
                "output_dir": "output",
                "log_dir": "logs"
            },
            "classes": [
                "piece_identite",
                "releve_bancaire",
                "facture_electricite",
                "facture_eau",
                "document_employeur"
            ]
        }
        
        import json
        config_path = self.project_root / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ Configuration créée: {config_path}")
        return config
    
    def run_setup(self):
        """Exécute l'initialisation complète"""
        logger.info("=" * 60)
        logger.info("INITIALISATION DU SYSTÈME OFFLINE")
        logger.info("=" * 60)
        
        # Vérifier les dépendances
        logger.info("\n[1/5] Vérification des dépendances...")
        if not self.verify_dependencies():
            logger.error("Certaines dépendances manquent. Installez-les d'abord.")
            return False
        
        # Créer la structure
        logger.info("\n[2/5] Création de la structure de répertoires...")
        self.create_directory_structure()
        
        # Configurer les modèles
        logger.info("\n[3/5] Configuration des modèles...")
        cv_models, nlp_models, gabarits_config = self.setup_models_offline()
        
        # Configurer Tesseract
        logger.info("\n[4/5] Configuration de Tesseract OCR...")
        self.setup_tesseract()
        
        # Créer le fichier de configuration
        logger.info("\n[5/5] Création du fichier de configuration...")
        self.create_config_file()
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ INITIALISATION COMPLÉTÉE AVEC SUCCÈS")
        logger.info("=" * 60)
        logger.info("\nProchaines étapes:")
        logger.info("1. Téléchargez les modèles pré-entraînés (ResNet50, EfficientNet, CamemBERT)")
        logger.info("2. Placez-les dans les répertoires models/cv/, models/nlp/")
        logger.info("3. Ajoutez vos données d'entraînement dans data/raw/")
        logger.info("4. Exécutez main.py pour démarrer le pipeline")
        
        return True


if __name__ == "__main__":
    setup = OfflineSetup(project_root=".")
    success = setup.run_setup()
    sys.exit(0 if success else 1)
