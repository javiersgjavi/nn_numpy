import numpy as np
import torch
from torch import nn

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

class IrisDataset:
    def __init__(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target

        self.y = np.where(self.y == 0, 0, 1)
        idx = np.arange(self.X.shape[0])
        np.random.shuffle(idx)

        test_idx = int(0.75 * self.X.shape[0])

        self.x_train = self.X[idx[:test_idx]]
        self.y_train = self.y[idx[:test_idx]]

        self.x_test = self.X[idx[test_idx:]]
        self.y_test = self.y[idx[test_idx:]]

        self.scaler = MinMaxScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    def get_train_data(self):
        return torch.tensor(self.x_train).float(), torch.tensor(self.y_train).float().view(-1, 1)
    

    def get_test_data(self):
        return torch.tensor(self.x_test).float(), torch.tensor(self.y_test).float().view(-1, 1)
    

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
def main():
    model = NN()
    dataset = IrisDataset()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x_train, y_train = dataset.get_train_data()


    for epoch in range(1000):
        x_train, y_train = dataset.get_train_data()

        # zero grad the gradients every batch
        optimizer.zero_grad()

        # forward pass
        y_pred = model(x_train)

        # calculate loss

        loss = criterion(y_pred, y_train)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {np.round(loss.item(), 4)}')

    x_test, y_test = dataset.get_test_data()

    y_pred = model(x_test)

    y_pred = np.where(y_pred.detach().numpy() > 0.5, 1, 0)

    accuracy = np.mean(y_pred == y_test.numpy())

    print(f'Accuracy: {accuracy}')

if __name__=='__main__':
    main()