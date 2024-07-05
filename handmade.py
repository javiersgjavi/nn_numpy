import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

def bce_loss(y_true, y_pred, epsilon=1e-15):

    # Agregar epsilon para evitar log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calcular la suma de la pérdida
    summatory = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Dividir por el número de ejemplos y tomar el signo negativo
    div = -1 / y_true.shape[0]
    return div * summatory

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
        self.y_train = self.y[idx[:test_idx]].reshape(-1, 1)

        self.x_test = self.X[idx[test_idx:]]
        self.y_test = self.y[idx[test_idx:]].reshape(-1, 1)

        self.scaler = MinMaxScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

    def get_train_data(self):
        return self.x_train, self.y_train
    
    def get_test_data(self):
        return self.x_test, self.y_test
    
class NN:
    def __init__(self):

        self.weights = {
            'W1': np.random.randn(4, 2),
            'W2': np.random.randn(2, 1),
            'b1': np.random.randn(1, 2),
            'b2': np.random.randn(1, 1)
        }

        self.gradients = {
            'W1': np.zeros_like(self.weights['W1']),
            'W2': np.zeros_like(self.weights['W2']),
            'b1': np.zeros_like(self.weights['b1']),
            'b2': np.zeros_like(self.weights['b2']),
        }

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def d_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def reset_grads(self):
        for key in self.gradients:
            self.gradients[key] = np.zeros_like(self.gradients[key])
    
    def forward(self, x):
        self.z1 = np.dot(x, self.weights['W1']) + self.weights['b1']
        self.h1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.h1, self.weights['W2']) + self.weights['b2']
        self.h2 = self.sigmoid(self.z2)
        return self.h2
    
    def backward(self, x, y_pred, y_true):
        m = y_true.shape[0]

        # gradiente respecto salida
        d_loss_h2 = y_pred - y_true.reshape(-1, 1)

        # gradiente para w2:
        self.gradients['W2'] = 1/m * np.dot(self.h1.T, d_loss_h2)
        self.gradients['b2'] = 1/m * np.sum(d_loss_h2, axis=0, keepdims=True)

        # gradientes capa oculta
        d_loss_h1 = np.dot(d_loss_h2, self.weights['W2'].T) * self.d_sigmoid(self.z1)

        # gradiente para w1
        self.gradients['W1'] = 1/m * np.dot(x.T, d_loss_h1)
        self.gradients['b1'] = 1/m * np.sum(d_loss_h1, axis=0, keepdims=True)

class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.01):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0
        self.lr = lr

        # momentums
        self.m = {
            'W1': 0,
            'W2': 0,
            'b1': 0,
            'b2': 0
        }

        # RMSprop
        self.v = {
            'W1': 0,
            'W2': 0,
            'b1': 0,
            'b2': 0
        }

    def momentum(self, g, m):
        m = self.beta1 * m + (1-self.beta1) * g
        return m
    
    def rmsprop(self, g, v):
        v = self.beta2*v + (1-self.beta2)*g**2
        return v
    
    def compute_final_values(self, m, v):
        m = m / (1 - self.beta1**self.t)
        v = v / (1 - self.beta2**self.t)
        return m, v
    
    def apply_gradients(self, m, v, lr, w):
        return w - lr*m/(np.sqrt(v) + self.epsilon)

    def update(self, weights, gradients):

        self.t +=1

        for weight in gradients:
            self.m[weight] = self.momentum(gradients[weight], self.m[weight]) # compute biased momentum
            self.v[weight] = self.rmsprop(gradients[weight], self.v[weight]) # compute biased RMSprop

            unbiased_m = self.m[weight] / (1 - self.beta1**self.t)
            unbiased_v = self.v[weight] / (1 - self.beta2**self.t)

            weights[weight] -= self.lr*unbiased_m/(np.sqrt(unbiased_v) + self.epsilon)

        return weights

def main():
    iris = IrisDataset()
    x_train, y_train = iris.get_train_data()
    x_test, y_test = iris.get_test_data()

    model = NN()
    optimizer = Adam()

    for epoch in range(1000):

        # Reseteamos los gradientes
        model.reset_grads()

        # Forward
        y_pred = model.forward(x_train)

        # Loss
        loss = bce_loss(y_train, y_pred)

        # Backward
        model.backward(x_train, y_pred, y_train)

        # Update
        new_weights = optimizer.update(model.weights, model.gradients)
        model.weights = new_weights

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    y_pred = model.forward(x_test)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy: {accuracy}')


if __name__=='__main__':
    main()
