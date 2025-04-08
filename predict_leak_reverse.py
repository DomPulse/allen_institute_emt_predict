import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # or use a classifier
from sklearn.metrics import mean_squared_error  # or accuracy_score for classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor

# Load your data
X = np.load("conduct_and_rest.npy")  # shape: (n_samples, n_features)
y = np.load("leaks.npy")  # shape: (n_samples,) or (n_samples, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and train a model
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, random_state=42)
#model = make_pipeline(StandardScaler(), SVR(C=30.0, epsilon=0.2))
# model = LogisticRegression()  # for classification
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
