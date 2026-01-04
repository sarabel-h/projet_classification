import os
import torch
from torch import nn, optim
from transformers import CamembertTokenizer
from torch.utils.data import Dataset, DataLoader

from ocr import extract_text
from nlp_model import CamembertClassifier, CLASS_NAMES

DATA_DIR = "data/raw"
MODEL_OUT = "models/camembert_classifier.pth"
EPOCHS = 3
BATCH_SIZE = 4
MAX_LEN = 512


# =========================
# Dataset NLP simple
# =========================
class OCRTextDataset(Dataset):
    def __init__(self, root_dir, tokenizer):
        self.samples = []
        self.tokenizer = tokenizer

        for label_idx, label in enumerate(CLASS_NAMES):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for file in os.listdir(label_dir):
                path = os.path.join(label_dir, file)
                self.samples.append((path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        text = extract_text(path)[:2000]

        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return (
            inputs["input_ids"].squeeze(0),
            inputs["attention_mask"].squeeze(0),
            torch.tensor(label)
        )


# =========================
# Entraînement
# =========================
def train():
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    dataset = OCRTextDataset(DATA_DIR, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CamembertClassifier()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for input_ids, attention_mask, labels in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print("CamemBERT finetuné et sauvegardé")


if __name__ == "__main__":
    train()
