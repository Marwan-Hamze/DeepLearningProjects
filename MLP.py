import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy, Precision, F1Score, ConfusionMatrix
import seaborn as sns

'''
In this example, I use a multi-layered perceptron (MLP) for a multi-class classification problem.
The "make_gaussian_quantiles" function from sklearn's datasets is used to generate the data.
Evaluation Metrics are used at the end to verify the accuracy of the model,
along with a plot of the confusion matrix.
'''

class MLP(nn.Module):
    def __init__(self, input_layer, hidden_layers, output_layer):
        super().__init__()
        self.sequence = nn.Sequential(nn.Linear(input_layer, hidden_layers),
                                      nn.ReLU(),
                                      nn.Linear(hidden_layers, hidden_layers),                                      
                                      nn.ReLU(),
                                      nn.Linear(hidden_layers, output_layer) 
                                      )
    def forward(self, x):
        return self.sequence(x)


n_samples = 2000
n_features = 3
n_classes = 5

X, Y = make_gaussian_quantiles(cov=3.0, n_samples=n_samples, n_features= n_features,
                                n_classes= n_classes, random_state= 42)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state= 42)

# Print the shapes

# print("Shape of X:", X.shape)
# print("Shape of Y:", Y.shape)
# print("Shape of x_train:", x_train.shape)
# print("Shape of y_test:", y_test.shape)

# Visualize the entire data
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection = "3d")
# ax.scatter(X[:,0], X[:,1], X[:,2], c = Y)
# ax.grid()
# plt.show()

# Transform the data to Tensors

x_train_tensor = torch.tensor(x_train, dtype= torch.float32)
x_test_tensor = torch.tensor(x_test, dtype= torch.float32)
y_train_tensor = torch.tensor(y_train, dtype= torch.long)
y_test_tensor = torch.tensor(y_test, dtype= torch.long)

# Initialize the model, loss, and optimizer

hidden_layers = 100

Model = MLP(n_features, hidden_layers, n_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params= Model.parameters(), lr= 0.1)

# Train the Model

epochs = 1000
for i in range(epochs):

    Model.train()
    output = Model(x_train_tensor)
    loss = loss_fn(output, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%100 == 0:
        print(f"Epoch = {i}, Loss = {loss}")

# Evaluate the Model

with torch.inference_mode():
    Model.eval()
    predictions_logits = Model(x_test_tensor)
    loss = loss_fn(predictions_logits, y_test_tensor)
    predictions = torch.softmax(predictions_logits, dim=1).argmax(dim=1)
    print("Loss during Inference: ", loss)

# Visualize the output and test data
# fig2 = plt.figure()
# ax = fig2.add_subplot(111, projection = "3d")
# ax.scatter(x_test[:,0], x_test[:,1], x_test[:,2], c = y_test)
# fig3 = plt.figure()
# ax = fig3.add_subplot(111, projection = "3d")
# ax.scatter(x_test[:,0], x_test[:,1], x_test[:,2], c = predictions)
# ax.grid()
# plt.show()

# Evaluation Metrics
accuracy = Accuracy(task="multiclass", num_classes= n_classes)
acc = accuracy(predictions, y_test_tensor)
precision = Precision(task="multiclass", num_classes= n_classes)
prec = precision(predictions, y_test_tensor)
f1_score = F1Score(task="multiclass", num_classes= n_classes)
score = f1_score(predictions, y_test_tensor)
confusion = ConfusionMatrix(task="multiclass", num_classes= n_classes)
conf = confusion(predictions, y_test_tensor)

print(f"Accuracy: {acc}| Precision: {prec}| F1Score: {score}")
print("Confusion Matrix:\n", conf)

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(torch.Tensor.numpy(conf), annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
