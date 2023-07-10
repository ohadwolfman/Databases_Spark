import numpy as np

# Load the data from CSV file
data = np.genfromtxt('apartment_prices.csv', delimiter=',')

# Split the data into features and target
X = data[:, :11]
y = data[:, 11]

# Add bias term to features
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

# Split the data into training and testing sets (75% for training, 25% for testing)
train_size = int(0.75 * X.shape[0])
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Calculate the regression coefficients using the normal equation
coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Predict the target variable for the test set
y_pred = X_test @ coefficients

# Calculate the mean squared error
mse = np.mean((y_pred - y_test) ** 2)

# Print the mean squared error
print("Mean Squared Error:", mse)
