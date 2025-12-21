"""
Pipeline principal de classification de documents
Intègre tous les modules: preprocessing, CV, NLP, gabarits, fusion
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocumentClassificationPipeline:
    """Pipeline complet de classification de documents"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialise le pipeline
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.classes = self.config.get("classes", [])
        
        # Initialiser les modules
        from src.preprocessing.preprocessor import DocumentPreprocessor
        from src.gabarits.detector import GabaritDetector
        from src.fusion.multimodal_fusion import MultimodalFusion
        from src.offline_manager import OfflineModelManager
        from src.computer_vision.cv_classifier import CVClassifier
        from src.nlp.nlp_classifier import NLPClassifier
        
        self.preprocessor = DocumentPreprocessor(dpi=300)
        self.gabarit_detector = GabaritDetector()
        self.fusion = MultimodalFusion()
        self.model_manager = OfflineModelManager(models_dir="models")
        
        # Initialiser les classificateurs RÉELS
        self.cv_classifier = CVClassifier(
            model_path="models/cv/resnet50_pretrained.pth",
            num_classes=len(self.classes),
            device='cpu'
        )
        
        self.nlp_classifier = NLPClassifier(
            model_path="models/nlp/camembert-base",
            num_classes=len(self.classes),
            device='cpu'
        )
        
        logger.info("[OK] Pipeline initialisé avec modèles réels")
    
    def _load_config(self, config_path: str) -> Dict:
        """Charge la configuration"""
        if not Path(config_path).exists():
            logger.warning(f"Config non trouvée: {config_path}, utilisant les valeurs par défaut")
            return self._default_config()
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            "classes": [
                "piece_identite",
                "releve_bancaire",
                "facture_electricite",
                "facture_eau",
                "document_employeur"
            ],
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed"
            },
            "output": {
                "output_dir": "output"
            }
        }
    
    def process_document(self, document_path: str) -> Dict:
        """
        Traite un document complet
        
        Args:
            document_path: Chemin vers le document (PDF ou image)
            
        Returns:
            Dict avec résultats de classification
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Traitement: {document_path}")
        logger.info(f"{'='*60}")
        
        doc_path = Path(document_path)
        results = {
            "document": str(doc_path),
            "timestamp": datetime.now().isoformat(),
            "pages": []
        }
        
        # Étape 1: Conversion PDF -> images
        if doc_path.suffix.lower() == '.pdf':
            logger.info("[1/7] Conversion PDF -> images")
            images = self.preprocessor.convert_pdf_to_images(document_path)
            if not images:
                logger.error("Erreur lors de la conversion du PDF")
                return results
        else:
            logger.info("[1/7] Chargement de l'image")
            image = self.preprocessor.load_image(document_path)
            images = [image] if image is not None else []
        
        # Traiter chaque page
        for page_idx, image in enumerate(images):
            logger.info(f"\n--- Page {page_idx + 1}/{len(images)} ---")
            page_result = self._process_page(image, page_idx)
            results["pages"].append(page_result)
        
        # Résumé final
        results["summary"] = self._generate_summary(results["pages"])
        
        logger.info(f"\n[OK] Classification terminée")
        logger.info(f"Résultat final: {results['summary']['predicted_class']}")
        logger.info(f"Confiance: {results['summary']['confidence']:.2f}")
        
        return results
    
    def _process_page(self, image: np.ndarray, page_idx: int) -> Dict:
        """
        Traite une seule page
        
        Args:
            image: Image numpy
            page_idx: Index de la page
            
        Returns:
            Dict avec résultats de la page
        """
        page_result = {
            "page_index": page_idx,
            "image_shape": image.shape,
        }
        
        try:
            # Étape 2: Extraction des features gabarits
            logger.info("[2/7] Extraction features gabarits")
            gabarit_scores = self.gabarit_detector.compute_gabarit_score(image, "")
            page_result["gabarit_scores"] = gabarit_scores
            logger.info(f"[OK] Gabarits: {max(gabarit_scores, key=gabarit_scores.get)} ({max(gabarit_scores.values()):.2f})")
            
            # Étape 3: Prétraitement pour CV
            logger.info("[3/7] Prétraitement CV")
            image_cv = self.preprocessor.preprocess_for_cv(image, target_size=(224, 224))
            page_result["cv_preprocessed"] = True
            
            # Étape 4: Extraction OCR
            logger.info("[4/7] Extraction OCR")
            image_ocr = self.preprocessor.preprocess_for_ocr(image)
            extracted_text = self.preprocessor.extract_text_with_ocr(image_ocr, lang='fra')
            page_result["extracted_text"] = extracted_text[:500] if extracted_text else ""
            logger.info(f"[OK] Texte extrait: {len(extracted_text)} caractères")
            
            # Étape 5: Classification CV RÉELLE avec ResNet50
            logger.info("[5/7] Classification CV (ResNet50)")
            cv_predictions = self.cv_classifier.predict(image_cv, self.classes)
            page_result["cv_predictions"] = cv_predictions
            
            # Étape 6: Classification NLP RÉELLE avec CamemBERT
            logger.info("[6/7] Classification NLP (CamemBERT)")
            nlp_predictions, semantic_patterns = self.nlp_classifier.predict(extracted_text, self.classes)
            page_result["nlp_predictions"] = nlp_predictions
            page_result["semantic_patterns"] = semantic_patterns
            
            # Étape 7: Fusion multimodale
            logger.info("[7/7] Fusion multimodale")
            final_class, confidence, reason = self.fusion.fuse_predictions(
                cv_predictions,
                nlp_predictions,
                gabarit_scores,
                semantic_patterns
            )
            
            page_result["final_prediction"] = final_class
            page_result["confidence"] = confidence
            page_result["fusion_reason"] = reason
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la page: {e}")
            page_result["error"] = str(e)
        
        return page_result
    
    def _generate_summary(self, pages: List[Dict]) -> Dict:
        """Génère un résumé de tous les pages"""
        if not pages:
            return {
                "predicted_class": "à_verifier",
                "confidence": 0.0,
                "pages_processed": 0
            }
        
        # Prendre la prédiction avec la meilleure confiance
        best_page = max(pages, key=lambda p: p.get("confidence", 0))
        
        return {
            "predicted_class": best_page.get("final_prediction", "à_verifier"),
            "confidence": best_page.get("confidence", 0.0),
            "pages_processed": len(pages),
            "best_page_index": pages.index(best_page)
        }
    
    def save_results(self, results: Dict, output_dir: str = "output"):
        """Sauvegarde les résultats"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_path / f"classification_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Résultats sauvegardés: {result_file}")
        
        return result_file
    
    def process_batch(self, documents_dir: str) -> List[Dict]:
        """Traite un lot de documents"""
        doc_dir = Path(documents_dir)
        
        if not doc_dir.exists():
            logger.error(f"Répertoire non trouvé: {documents_dir}")
            return []
        
        documents = list(doc_dir.glob("**/*.pdf")) + list(doc_dir.glob("**/*.jpg")) + list(doc_dir.glob("**/*.png"))
        
        logger.info(f"Traitement de {len(documents)} documents...")
        
        all_results = []
        for doc in documents:
            result = self.process_document(str(doc))
            all_results.append(result)
        
        return all_results


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline de classification de documents")
    parser.add_argument("document", nargs="?", help="Chemin vers le document/répertoire")
    parser.add_argument("--config", default="config.json", help="Chemin vers le fichier de configuration")
    parser.add_argument("--output", default="output", help="Répertoire de sortie")
    
    args = parser.parse_args()
    
    if not args.document:
        logger.error("Veuillez spécifier un document ou répertoire")
        parser.print_help()
        sys.exit(1)
    
    # Initialiser le pipeline
    pipeline = DocumentClassificationPipeline(config_path=args.config)
    
    # Traiter le document/répertoire
    doc_path = Path(args.document)
    
    if doc_path.is_file():
        results = pipeline.process_document(str(doc_path))
        pipeline.save_results(results, args.output)
    
    elif doc_path.is_dir():
        results = pipeline.process_batch(str(doc_path))
        logger.info(f"\n[OK] {len(results)} documents traités")
        
        # Sauvegarder les résultats du batch
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_file = output_path / f"batch_classification_{timestamp}.json"
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[OK] Résultats batch sauvegardés: {batch_file}")
    
    else:
        logger.error(f"Fichier/répertoire non trouvé: {args.document}")
        sys.exit(1)


if __name__ == "__main__":
    main()
