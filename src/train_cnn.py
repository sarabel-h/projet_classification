import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

DATA_DIR = 'data/cnn_images'
MODEL_OUT = 'models/cnn_resnet50.pth'
os.makedirs('models', exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(2),
    transforms.ColorJitter(brightness=0.2, contrast=0.1),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# freeze backbone initially
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 5)
# only train fc first
for param in model.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

for epoch in range(3):
    for imgs, labels in loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# optional: unfreeze some layers (simple)
for param in list(model.parameters())[-20:]:
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(2):
    for imgs, labels in loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), MODEL_OUT)
print('CNN saved to', MODEL_OUT)
