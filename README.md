## Import libraries
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load numerical dataset
data = load_diabetes()
X, y = data.data, data.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Overfitting model (too complex)
overfit_model = MLPRegressor(hidden_layer_sizes=(200, 200), max_iter=2000, random_state=42)
overfit_model.fit(X_train, y_train)

# Model 2: Regularized model (simpler, avoids overfitting)
regular_model = MLPRegressor(hidden_layer_sizes=(20,), alpha=0.1, max_iter=1000, random_state=42)
regular_model.fit(X_train, y_train)

# Evaluate both models
y_pred_train_overfit = overfit_model.predict(X_train)
y_pred_test_overfit = overfit_model.predict(X_test)
y_pred_train_reg = regular_model.predict(X_train)
y_pred_test_reg = regular_model.predict(X_test)

# Compute errors
print("=== Overfitting Model ===")
print("Train MSE:", mean_squared_error(y_train, y_pred_train_overfit))
print("Test  MSE:", mean_squared_error(y_test, y_pred_test_overfit))

print("\n=== Regularized Model ===")
print("Train MSE:", mean_squared_error(y_train, y_pred_train_reg))
print("Test  MSE:", mean_squared_error(y_test, y_pred_test_reg))

