import torch
import torch.nn as nn
from torchvision import models
import os
CLASS_NAMES = [
    "piece_identite",
    "facture_electricite",
    "facture_eau",
    "releve_bancaire",
    "document_employeur"
]

def load_cnn_model(weights_path=None):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
        except Exception:
            pass
    model.eval()
    return model
