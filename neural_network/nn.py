import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x
    

def train_model(model, x, y, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    return model

#use test data to evaluate the model
def evaluate_model(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        criterion = nn.MSELoss()
        loss = criterion(y_pred, y)
        pred = (y_pred >= 0.5).float()
        accuracy = (pred == y).float().mean()

    return loss.item(), accuracy.item()