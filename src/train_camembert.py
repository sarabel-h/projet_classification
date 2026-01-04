import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer
from nlp_model import CamembertClassifier, CLASS_NAMES
from ocr import extract_text, split_text_lang
from sklearn.model_selection import train_test_split

DATA_DIR = "data/raw"
MODEL_OUT = "models/camembert_classifier.pth"
EPOCHS = 10
BATCH_SIZE = 4
MAX_LEN = 256

tokenizer = CamembertTokenizer.from_pretrained("camembert-base")


class OCRTextDataset(Dataset):
    def __init__(self, files, tokenizer):
        self.samples = files
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        text = extract_text(path)
        fr, ar = split_text_lang(text)
        text_use = fr if fr.strip() else text[:2000]
        inputs = self.tokenizer(text_use, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), torch.tensor(label_idx)


def collect_files(root):
    files = []
    for label_idx, label in enumerate(CLASS_NAMES):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        for f in os.listdir(label_dir):
            files.append((os.path.join(label_dir, f), label_idx))
    return files


def train():
    files = collect_files(DATA_DIR)
    if not files:
        print('No data found in data/raw. Add files first.')
        return
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)
    train_ds = OCRTextDataset(train_files, tokenizer)
    val_ds = OCRTextDataset(val_files, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = CamembertClassifier()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f'Epoch {epoch+1} loss={total:.4f}')

    torch.save(model.state_dict(), MODEL_OUT)
    print('Saved Camembert model to', MODEL_OUT)


if __name__ == '__main__':
    train()
