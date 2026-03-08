import numpy as np

class MiniBatchGradientDescent:

    def __init__(self, lr=0.01, epochs=200, batch_size=4):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features,1))
        self.bias = 0

        for _ in range(self.epochs):

            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            for i in range(0, n_samples, self.batch_size):

                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                y_pred = X_batch.dot(self.weights) + self.bias
                error = y_pred - y_batch

                loss = np.mean(error**2)
                self.loss_history.append(loss)

                dw = (1/len(X_batch)) * X_batch.T.dot(error)
                db = (1/len(X_batch)) * np.sum(error)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db