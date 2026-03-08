import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

from src.preprocessing import load_and_prepare_data
from src.batch_gd import BatchGradientDescent
from src.stochastic_gd import StochasticGradientDescent
from src.minibatch_gd import MiniBatchGradientDescent
from src.metrics import rmse

X_train, X_test, y_train, y_test = load_and_prepare_data("data/housing.csv")

bgd = BatchGradientDescent(lr=0.01, epochs=200)
sgd = StochasticGradientDescent(lr=0.01, epochs=30)
mbgd = MiniBatchGradientDescent(lr=0.01, epochs=200, batch_size=4)

bgd.fit(X_train, y_train)
sgd.fit(X_train, y_train)
mbgd.fit(X_train, y_train)

bgd_pred = X_test.dot(bgd.weights) + bgd.bias
sgd_pred = X_test.dot(sgd.weights) + sgd.bias
mbgd_pred = X_test.dot(mbgd.weights) + mbgd.bias

sk_model = SGDRegressor(max_iter=1000)
sk_model.fit(X_train, y_train.ravel())
sk_pred = sk_model.predict(X_test)

print("Batch GD RMSE:", rmse(y_test, bgd_pred))
print("SGD RMSE:", rmse(y_test, sgd_pred))
print("MiniBatch GD RMSE:", rmse(y_test, mbgd_pred))
print("Sklearn SGD RMSE:", rmse(y_test, sk_pred))

plt.figure(figsize=(8,5))
plt.plot(bgd.loss_history, label="Batch GD")
plt.plot(sgd.loss_history, label="SGD")
plt.plot(mbgd.loss_history, label="MiniBatch GD")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Gradient Descent Convergence")
plt.legend()
plt.savefig("plots/convergence_plot.png")
plt.show()