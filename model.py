import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Simulate data
np.random.seed(42)
num_samples = 1000
num_features = 50
X = np.random.rand(num_samples, num_features)
coefficients = np.random.rand(num_features)
y = X.dot(coefficients) + np.random.randn(num_samples) * 0.1
feature_names = [f'Feature_{i}' for i in range(num_features)]
data = pd.DataFrame(X, columns=feature_names)
data['Age'] = y

# Preprocess the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Split the data
X = data_scaled.drop('Age', axis=1)
y = data_scaled['Age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
