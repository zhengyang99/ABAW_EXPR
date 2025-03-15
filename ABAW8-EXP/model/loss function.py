import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class CustomDataset(Dataset):
    def __init__(self, num_samples, input_size, num_classes_coarse):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes_coarse = num_classes_coarse
        self.data = torch.randn(num_samples, input_size).float()
        self.labels_coarse = torch.randint(0, num_classes_coarse, (num_samples,)).long()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels_coarse[idx]

class ComplexModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes_coarse):
        super(ComplexModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes_coarse)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}, Loss: {average_loss}')

def main():
    input_size = 100
    hidden_size = 128
    num_classes_coarse = 5
    num_samples = 1000
    batch_size = 32
    num_epochs = 10

    dataset = CustomDataset(num_samples, input_size, num_classes_coarse)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ComplexModel(input_size, hidden_size, num_classes_coarse)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    train_model(model, dataloader, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    main()