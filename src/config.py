
import os

# --- Classes ---
CLASS_NAMES = [
    "piece_identite",
    "facture_electricite",
    "facture_eau",
    "releve_bancaire",
    "document_employeur"
]

# --- Chemins ---
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed_images"
MODELS_DIR = "models"
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_resnet50.pth")
NLP_MODEL_PATH = os.path.join(MODELS_DIR, "camembert_classifier.pth")

# --- Paramètres ---
IMG_SIZE = 224
MAX_LEN_NLP = 256
BATCH_SIZE = 8
EPOCHS = 10

# --- Seuils de Fusion ---
# Si le score combiné est inférieur à ça, c'est 'unknown'
THRESHOLD_UNKNOWN = 0.40 
