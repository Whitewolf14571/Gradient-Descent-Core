import numpy as np

class StochasticGradientDescent:

    def __init__(self, lr=0.01, epochs=30):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features,1))
        self.bias = 0

        for _ in range(self.epochs):

            for i in range(n_samples):

                idx = np.random.randint(n_samples)

                X_i = X[idx:idx+1]
                y_i = y[idx:idx+1]

                y_pred = X_i.dot(self.weights) + self.bias
                error = y_pred - y_i

                loss = np.mean(error**2)
                self.loss_history.append(loss)

                dw = X_i.T.dot(error)
                db = error

                self.weights -= self.lr * dw
                self.bias -= self.lr * db