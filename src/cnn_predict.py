import torch
from .cnn_model import load_cnn_model, CLASS_NAMES
from .image_utils import load_image

MODEL_PATH = "models/cnn_resnet50.pth"

model = load_cnn_model(MODEL_PATH)

def predict_image(image_path):
    img = load_image(image_path)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]

    score, idx = torch.max(probs, dim=0)
    return CLASS_NAMES[idx], round(score.item(), 2)
