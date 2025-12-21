"""
Module de classification par Computer Vision
Utilise ResNet50 pré-entraîné pour classifier les documents
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class CVClassifier:
    """Classificateur basé sur ResNet50"""
    
    def __init__(self, model_path: str, num_classes: int = 5, device: str = 'cpu'):
        """
        Initialise le classificateur CV
        
        Args:
            model_path: Chemin vers le modèle ResNet50
            num_classes: Nombre de classes
            device: 'cpu' ou 'cuda'
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        self.model = self._load_model(model_path)
        self.model.eval()
        
        logger.info(f"[OK] CVClassifier initialisé sur {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Charge le modèle ResNet50"""
        model = models.resnet50(pretrained=False)
        
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        
        if Path(model_path).exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Modèle chargé depuis: {model_path}")
            except Exception as e:
                logger.warning(f"Impossible de charger les poids: {e}")
                logger.info("Utilisation du modèle pré-entraîné ImageNet")
                
                model = models.resnet50(pretrained=True)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, self.num_classes)
        else:
            logger.warning(f"Modèle non trouvé: {model_path}")
            logger.info("Utilisation du modèle pré-entraîné ImageNet")
            
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        
        return model.to(self.device)
    
    def predict(self, image: np.ndarray, class_names: list) -> Dict[str, float]:
        """
        Prédiction sur une image
        
        Args:
            image: Image prétraitée (224x224x3), normalisée [0, 1]
            class_names: Liste des noms de classes
            
        Returns:
            Dict {classe: probabilité}
        """
        if image.shape != (224, 224, 3):
            logger.error(f"Mauvaise forme d'image: {image.shape}, attendu (224, 224, 3)")
            return {cls: 1.0 / len(class_names) for cls in class_names}
        
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()[0]
        
        predictions = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        logger.info(f"Prédiction CV: {max(predictions, key=predictions.get)} ({max(predictions.values()):.3f})")
        
        return predictions
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extrait les features d'une image
        
        Args:
            image: Image prétraitée
            
        Returns:
            Vecteur de features
        """
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            features = self.model.avgpool(self.model.layer4(
                self.model.layer3(self.model.layer2(
                    self.model.layer1(self.model.maxpool(
                        self.model.relu(self.model.bn1(self.model.conv1(image_tensor)))
                    ))
                ))
            ))
            features = torch.flatten(features, 1)
            features = features.cpu().numpy()[0]
        
        return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    classifier = CVClassifier(
        model_path="models/cv/resnet50_pretrained.pth",
        num_classes=5
    )
    
    test_image = np.random.rand(224, 224, 3)
    class_names = ["piece_identite", "releve_bancaire", "facture_electricite", "facture_eau", "document_employeur"]
    
    predictions = classifier.predict(test_image, class_names)
    print("Prédictions:", predictions)
