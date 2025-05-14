import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

'''
In this example, I use  Long Short-Term Memory (LSTM) network to predict the next value in a time series.
'''

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out[:, -1])
        return out

# Generate a sine wave time series
time_steps = np.linspace(0, 100, 1000)
data = np.sin(time_steps)
# plt.plot(time_steps, data)
# plt.title("Sine Wave Time Series")
# plt.show()

# print("DATA: ", data.shape)

# Hyperparameters
seq_length = 10
X, y = [], []

# Create sequences: In the input X, we have 10 values, 
# and in the output y, we have the value that follows those in X 
for i in range(len(data) - seq_length):
    X.append(data[i: i + seq_length])
    y.append(data[i + seq_length])

# print("LIST: ", len(X))

X = np.array(X)
y = np.array(y)

# print("NUMPY: ", X.shape)
# Convert to tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # [batch, seq_length, features]
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# print("TENSOR: ", X.size())

# Initializing the Model, Loss, and Optimizer
model = LSTM(input_dim=1, hidden_dim=50, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation/ Inference
model.eval()
with torch.inference_mode():
    predictions = model(X).detach().numpy()

plt.plot(time_steps[seq_length:], data[seq_length:], label="Original Data")
plt.plot(time_steps[seq_length:], predictions, label="LSTM Predictions")
plt.legend()
plt.show()

