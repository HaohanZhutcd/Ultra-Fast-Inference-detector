import argparse
import os
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F

from model.RepVGGLSTM import LSTM_ResNet18,RepVGGFeatureExtractor
from utils.dataset import cropDataset
from utils.general import check_and_create_folder, increment_path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
checkpoints_dir = Path(ROOT / 'checkpoints')
checkpoints_dir.mkdir(parents=True, exist_ok=True)
save_dir = increment_path(Path(checkpoints_dir / 'exp'), exist_ok= False, mkdir = True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = cropDataset(ROOT/'data/crop_imgs', transform=transform, time_step= 16)
val_dataset = cropDataset(ROOT/'data/crop_imgs_val', transform=transform, time_step= 16)

model = RepVGGFeatureExtractor(hidden_size=256, num_classes=8)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

best_accuracy = 0.0
best_model_state_dict = None
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_losses = []
val_accuracies = []
all_labels = []
all_predictions = []


epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    train_size = len(train_loader.dataset)

    for batch, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        pred = model(inputs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(inputs)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")
        # =================== end-batch =============================
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
    #======================== end-epoch ================================
    # ============================ end-training ================================


    model.eval()
    test_loss, correct = 0, 0
    correct = 0
    total = 0
    val_size = len(val_loader.dataset)
    num_Valbatches = len(val_loader)
    with torch.no_grad():
        for inputs, labels in val_loader:
            # inputs = inputs.to(device, non_blocking=True).float() / 255
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = model(inputs)

            test_loss += criterion(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            # all_labels.extend(labels.cpu().numpy().astype(int))
            # all_predictions.extend(pred.cpu().numpy().astype(int))

    test_loss /= num_Valbatches
    correct /= val_size
    accuracy = 100 * correct
    val_accuracies.append(accuracy)

    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # accuracy = 100 * correct / total
    # print(f"Validation Accuracy: {accuracy:.2f}%")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state_dict = model.state_dict()

torch.save({'model_state_dict':model.state_dict()}, str(save_dir / 'last.pt'))
if best_model_state_dict is not None:
    torch.save({'model_state_dict':best_model_state_dict}, str(save_dir / 'best.pt'))

plt.plot(train_losses, label='Training loss')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.legend()
plt.savefig(save_dir / 'loss.png')
# plt.show()

