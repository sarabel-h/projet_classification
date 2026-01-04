import torch
import torch.nn as nn
from transformers import CamembertModel, CamembertTokenizer

CLASS_NAMES = [
    "piece_identite",
    "facture_electricite",
    "facture_eau",
    "releve_bancaire",
    "document_employeur"
]

class CamembertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CamembertModel.from_pretrained("camembert-base")
        self.fc = nn.Linear(768, len(CLASS_NAMES))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls)
