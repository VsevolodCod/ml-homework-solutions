import json
import os
import re
import numpy as np
import torch
import torchvision
from IPython.display import clear_output
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST

# Функции для оценки модели (предоставлены в задании)
def get_predictions(model, eval_data, step=10):
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for idx in range(0, len(eval_data), step):
            y_predicted = model(eval_data[idx : idx + step].to(device))
            predicted_labels.append(y_predicted.argmax(dim=1).cpu())
    predicted_labels = torch.cat(predicted_labels)
    predicted_labels = ",".join([str(x.item()) for x in list(predicted_labels)])
    return predicted_labels

def get_accuracy(model, data_loader):
    predicted_labels = []
    real_labels = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            y_predicted = model(batch[0].to(device))
            predicted_labels.append(y_predicted.argmax(dim=1).cpu())
            real_labels.append(batch[1])
    predicted_labels = torch.cat(predicted_labels)
    real_labels = torch.cat(real_labels)
    accuracy_score = (predicted_labels == real_labels).type(torch.FloatTensor).mean()
    return accuracy_score

# Настройка устройства
CUDA_DEVICE_ID = 0
device = torch.device(f"cuda:{CUDA_DEVICE_ID}") if torch.cuda.is_available() else torch.device("cpu")

# Загрузка данных
train_fmnist_data = FashionMNIST(
    ".", train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_fmnist_data = FashionMNIST(
    ".", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

train_data_loader = torch.utils.data.DataLoader(
    train_fmnist_data, batch_size=32, shuffle=True, num_workers=0
)
test_data_loader = torch.utils.data.DataLoader(
    test_fmnist_data, batch_size=32, shuffle=False, num_workers=0
)

# Определение модели
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Создание модели
model_task_1 = FashionMNISTModel()
model_task_1.to(device)

# Функция обучения
def train_model(model, train_loader, test_loader, num_epochs=15, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 500 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Оценка на тестовой выборке
        test_acc = get_accuracy(model, test_loader)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
    return train_losses, test_accuracies

if __name__ == '__main__':
    # Обучение модели
    train_losses, test_accuracies = train_model(model_task_1, train_data_loader, test_data_loader)

    # Оценка качества
    train_acc_task_1 = get_accuracy(model_task_1, train_data_loader)
    print(f"Neural network accuracy on train set: {train_acc_task_1:.5f}")
    test_acc_task_1 = get_accuracy(model_task_1, test_data_loader)
    print(f"Neural network accuracy on test set: {test_acc_task_1:.5f}")

    # Проверка порогов
    if test_acc_task_1 >= 0.885:
        print("✅ Test accuracy threshold (0.885) passed!")
    else:
        print(f"❌ Test accuracy {test_acc_task_1:.5f} is below 0.885 threshold")
    
    if train_acc_task_1 >= 0.905:
        print("✅ Train accuracy threshold (0.905) passed!")
    else:
        print(f"❌ Train accuracy {train_acc_task_1:.5f} is below 0.905 threshold")

    # Генерация файла для отправки (опционально)
    try:
        loaded_data_dict = np.load("hw_fmnist_data_dict.npy", allow_pickle=True)
        
        submission_dict = {
            "train_predictions_task_1": get_predictions(
                model_task_1, torch.FloatTensor(loaded_data_dict.item()["train"])
            ),
            "test_predictions_task_1": get_predictions(
                model_task_1, torch.FloatTensor(loaded_data_dict.item()["test"])
            ),
        }

        with open("submission_dict_fmnist_task_1.json", "w") as iofile:
            json.dump(submission_dict, iofile)
        print("✅ File saved to `submission_dict_fmnist_task_1.json`")
        
    except FileNotFoundError:
        print("⚠️  File 'hw_fmnist_data_dict.npy' not found. Skipping submission generation.")
        print("   If you need to generate submission, make sure this file is in the current directory.")