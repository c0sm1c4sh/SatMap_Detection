import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader

# --- Fix Python Path ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from model_utils import get_model, get_transform, get_device
except ImportError as e:
    print(f"❌ Error: {e}. Check your backend/model_utils.py file.")
    sys.exit(1)

# 1. Setup GPU
device = get_device()
print(f"🚀 Training on: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

CLASSES = ['Forest', 'HerbaceousVegetation', 'PermanentCrop', 'Residential', 'Industrial']
BATCH_SIZE = 64 

# 2. Data Loading
transform = get_transform()
full_dataset = datasets.EuroSAT(root="./data", download=True, transform=transform)

# Filter dataset to our 5 classes
idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
relevant_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if idx_to_class[label] in CLASSES]
subset = torch.utils.data.Subset(full_dataset, relevant_indices)
train_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Model
model = get_model(num_classes=5)
model = model.to(device) # Force move to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
print(f"Starting training for {len(subset)} images...")
model.train()
for epoch in range(5):
    running_loss = 0.0
    for images, labels in train_loader:
        # Move both data and targets to GPU
        images = images.to(device)
        labels_mapped = torch.tensor([CLASSES.index(idx_to_class[l.item()]) for l in labels]).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_mapped)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/5 | Loss: {running_loss/len(train_loader):.4f}")

# 5. Save Weights
os.makedirs('backend/weights', exist_ok=True)
torch.save(model.state_dict(), 'backend/weights/model.pth')
print("✅ Done! Weights saved to backend/weights/model.pth")