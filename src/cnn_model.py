import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
import os

CLASS_NAMES = [
    "piece_identite",
    "facture_electricite",
    "facture_eau",
    "releve_bancaire",
    "document_employeur"
]

def load_cnn_model(weights_path=None):
    # Charger ResNet50 avec poids ImageNet (sans fc custom)
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Remplacer la tête
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    # Charger UNIQUEMENT un modèle finetuné sur 5 classes
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)

    model.eval()
    return model
