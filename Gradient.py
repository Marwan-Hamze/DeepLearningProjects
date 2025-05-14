import numpy as np
import matplotlib.pyplot as plt

'''
In this file, a simple gradient descent method is implemented, to fit random data
into a linear regression model.
'''

# Generate Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1) # rand gives a number between 0 and 1, (100, 1) is the shape of X.
y = 3 * X + 4 + np.random.randn(100, 1)/2  # y = 3x + 4 + noise. Noise is for additional difficulty.

# Initialize Parameters
# randn gives a number from a normal distribution
w = np.random.randn(1, 1)  # Random initial weight
b = np.random.randn(1, 1)  # Random initial bias
learning_rate = 0.01
epochs = 300

# Gradient Descent Loop
for epoch in range(epochs):
    # Forward Pass: Compute predictions
    y_pred = X @ w + b  # Matrix multiplication for prediction

    # Compute Loss (Mean Squared Error)
    loss = (1 / len(X)) * np.sum((y_pred - y) ** 2)

    # Backward Pass: Compute gradients
    dw = (2 / len(X)) * X.T @ (y_pred - y)
    db = (2 / len(X)) * np.sum(y_pred - y)

    # Update Parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # Print loss every 100 iterations
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Plot the results
plt.scatter(X, y, label="Data Points")
plt.plot(X, X @ w + b, color='red', label="Fitted Line")
plt.legend()
plt.show()
