
import os
# Désactive le parallélisme des tokenizers pour éviter le conflit mémoire
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from transformers import CamembertTokenizer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import shutil
from src.config import *
from src.models_arch import CamembertClassifier, get_cnn_model
from src.ocr import extract_text_from_image, split_languages
from src.preprocessing import convert_pdf_to_images
from collections import Counter

# ================================
# 1. Préparation des données CNN
# ================================
def prepare_cnn_dataset():
    print(">>> Préparation des images pour CNN...")
    if not os.path.exists(DATA_RAW):
        print(f"Erreur: Dossier {DATA_RAW} introuvable.")
        return

    # On nettoie le dossier processed
    if os.path.exists(DATA_PROCESSED):
        shutil.rmtree(DATA_PROCESSED)
    
    os.makedirs(DATA_PROCESSED)
    
    # Parcours des classes
    for cls in CLASS_NAMES:
        raw_cls_dir = os.path.join(DATA_RAW, cls)
        proc_cls_dir = os.path.join(DATA_PROCESSED, cls)
        os.makedirs(proc_cls_dir, exist_ok=True)
        
        if not os.path.exists(raw_cls_dir): continue
        
        files = os.listdir(raw_cls_dir)
        count = 0
        for f in files:
            fpath = os.path.join(raw_cls_dir, f)
            # Si PDF -> Split
            if f.lower().endswith('.pdf'):
                imgs = convert_pdf_to_images(fpath, output_folder=proc_cls_dir)
                count += len(imgs)
            # Si Image -> Copy
            elif f.lower().endswith(('.jpg', '.png', '.jpeg')):
                shutil.copy(fpath, proc_cls_dir)
                count += 1
        print(f"Classe {cls}: {count} images préparées.")

# ================================
# 2. Entraînement CNN
# ================================
def train_cnn(device):
    print("\n>>> Démarrage entraînement CNN...")
    
    # Augmentation pour combler le manque de données
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10), # Rotation légère
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Robustesse scans
        transforms.RandomHorizontalFlip(p=0.1), # Rare pour des docs mais possible
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(DATA_PROCESSED)
    
    # Split Stratifié (Indispensable pour petit dataset déséquilibré)
    # On utilise les index pour splitter
    indices = list(range(len(full_dataset)))
    labels = [y for _, y in full_dataset.samples]
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(indices, labels))
    
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    
    # On applique les transforms dynamiquement (c'est une simplification ici,
    # idéalement on aurait deux datasets wrapper distincts, mais pour simplifier
    # on applique train_transforms au dataset global, validation sera un peu augmentée aussi
    # ce qui n'est pas grave pour un proto, ou on duplique le dataset).
    # Pour faire propre :
    train_dataset = datasets.ImageFolder(DATA_PROCESSED, transform=train_transforms)
    val_dataset = datasets.ImageFolder(DATA_PROCESSED, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_subsampler)
    
    model = get_cnn_model(len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), CNN_MODEL_PATH)
            
    print(f"Meilleur CNN sauvegardé avec Acc: {best_acc:.2f}%")

# ================================
# 3. Entraînement NLP
# ================================
class TextDataset(Dataset):
    def __init__(self, data_list, tokenizer):
        self.data = data_list # List of (text, label_idx)
        self.tokenizer = tokenizer
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LEN_NLP, return_tensors="pt")
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), torch.tensor(label)

def train_nlp(device):
    print("\n>>> Démarrage entraînement NLP...")
    
    # Configuration Tesseract (Optionnel, à décommenter sur Windows si besoin)
    # import pytesseract
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # 1. Extraction du texte
    texts = []
    labels = []
    
    for label_idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(DATA_PROCESSED, cls)
        if not os.path.exists(folder): continue
        files = os.listdir(folder)
        
        count_ok = 0 
        print(f"   Traitement OCR pour {cls} ({len(files)} fichiers)...")
        
        for f in files:
            path = os.path.join(folder, f)
            try:
                txt = extract_text_from_image(path)
                fr, _ = split_languages(txt)
                content = fr if len(fr) > 10 else txt 
                
                if content and len(content.strip()) > 5:
                    texts.append(content)
                    labels.append(label_idx)
                    count_ok += 1
            except Exception as e:
                print(f"      Erreur lecture {f}: {e}")

        print(f"      -> {count_ok} textes valides extraits.")

    if not texts:
        print("ERREUR CRITIQUE: Aucun texte extrait.")
        return
    
    print(f"Distribution des données NLP : {Counter(labels)}")

    # 2. Séparation des données
    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, 
            test_size=0.2, 
            random_state=42, 
            shuffle=True
        )
    except ValueError:
        print(f"Attention: Trop peu de données. Tout est utilisé pour l'entraînement.")
        train_texts, train_labels = texts, labels
        val_texts, val_labels = [], []

    print(f"Données: {len(train_texts)} train, {len(val_texts)} validation")

    # Création des datasets (CORRECTION ICI : list(zip(...)))
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    
    # On convertit le zip en liste pour que len() fonctionne
    train_ds = TextDataset(list(zip(train_texts, train_labels)), tokenizer) 
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    
    val_loader = None
    if val_texts:
        # Idem ici : list(zip(...))
        val_ds = TextDataset(list(zip(val_texts, val_labels)), tokenizer)
        val_loader = DataLoader(val_ds, batch_size=4)
    
    model = CamembertClassifier(len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    best_acc = 0.0
    
    for epoch in range(5): 
        model.train()
        total_loss = 0
        for ids, mask, lbl in train_loader:
            ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
            optimizer.zero_grad()
            outputs = model(ids, mask)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_acc_str = "N/A"
        if val_loader:
            model.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for ids, mask, lbl in val_loader:
                    ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
                    outputs = model(ids, mask)
                    _, predicted = torch.max(outputs, 1)
                    total += lbl.size(0)
                    correct += (predicted == lbl).sum().item()
            
            if total > 0:
                acc = 100 * correct / total
                val_acc_str = f"{acc:.2f}%"
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), NLP_MODEL_PATH)
            else:
                torch.save(model.state_dict(), NLP_MODEL_PATH)
        else:
            torch.save(model.state_dict(), NLP_MODEL_PATH)
        
        print(f"NLP Epoch {epoch+1} - Loss: {total_loss:.4f} - Val Acc: {val_acc_str}")
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")
    
    prepare_cnn_dataset()
    #train_cnn(device)
    train_nlp(device)
