import numpy as np

class BatchGradientDescent:

    def __init__(self, lr=0.01, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features,1))
        self.bias = 0

        for _ in range(self.epochs):

            y_pred = X.dot(self.weights) + self.bias
            error = y_pred - y

            loss = np.mean(error**2)
            self.loss_history.append(loss)

            dw = (1/n_samples) * X.T.dot(error)
            db = (1/n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db