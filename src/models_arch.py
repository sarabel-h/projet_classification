
import torch.nn as nn
from transformers import CamembertModel
from torchvision import models

class CamembertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = CamembertModel.from_pretrained("camembert-base")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # On utilise le token [CLS] (index 0)
        cls_token = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_token)
        return self.fc(x)

def get_cnn_model(num_classes, pretrained=True):
    # Utilisation de poids par défaut (ImageNet)
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Remplacer la dernière couche FC
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
