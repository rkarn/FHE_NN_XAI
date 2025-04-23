import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from concrete.ml.torch.compile import compile_torch_model

# Define the fully connected neural network architecture without Softmax
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=250, output_size=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # No Softmax, returns raw logits

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)

# Use a subset of the dataset for faster training
train_size = 10000  # Use only 10k samples for training
test_size = 1000    # Use only 1k samples for testing
mnist_train, _ = random_split(mnist_train, [train_size, len(mnist_train) - train_size])
mnist_test, _ = random_split(mnist_test, [test_size, len(mnist_test) - test_size])

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Initialize the neural network
input_size = 784  # 28x28 images flattened
hidden_size = 250
output_size = 10
model = SimpleNN(input_size, hidden_size, output_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss works with raw logits
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.view(-1, 28*28)  # Flatten images to 1D vectors
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Compile the trained PyTorch model for FHE
inputset = torch.zeros((1, 784), dtype=torch.float32)  # Dummy input for tracing the model
quantized_model = compile_torch_model(model, inputset)

# Test encrypted inference on a single test sample
test_sample, label = mnist_test[0]
test_sample = test_sample.view(-1, 28*28).numpy()

# Perform encrypted inference
encrypted_result = quantized_model.forward(test_sample)

# Manually find the predicted class (argmax of logits)
predicted_class = np.argmax(encrypted_result)

print(f"Predicted (encrypted) label: {predicted_class}")
print(f"True label: {label}")


import numpy as np
import torch

# Function to evaluate accuracy
def evaluate_accuracy(model, data_loader):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Accuracy of the unencrypted model
unencrypted_accuracy = evaluate_accuracy(model, test_loader)
print(f"Accuracy of the unencrypted model: {unencrypted_accuracy:.2f}%")

# Accuracy of the encrypted model
correct_encrypted = 0
total_encrypted = 0
for images, labels in test_loader:
    images = images.view(-1, 28*28).numpy()
    encrypted_outputs = quantized_model.forward(images)
    predicted_encrypted = np.argmax(encrypted_outputs, axis=1)
    correct_encrypted += np.sum(predicted_encrypted == labels.numpy())
    total_encrypted += labels.size(0)

encrypted_accuracy = 100 * correct_encrypted / total_encrypted
print(f"Accuracy of the encrypted model: {encrypted_accuracy:.2f}%")


# Preprocess data consistently
def preprocess_data(images):
    return images.view(-1, 28*28).numpy()  # Flatten images to 1D vectors

# Evaluate unencrypted model accuracy
unencrypted_accuracy = evaluate_accuracy(model, test_loader)
print(f"Accuracy of the unencrypted model: {unencrypted_accuracy:.2f}%")

# Evaluate encrypted model accuracy
correct_encrypted = 0
total_encrypted = 0
for images, labels in test_loader:
    preprocessed_images = preprocess_data(images)
    encrypted_outputs = quantized_model.forward(preprocessed_images)
    predicted_encrypted = np.argmax(encrypted_outputs, axis=1)
    correct_encrypted += np.sum(predicted_encrypted == labels.numpy())
    total_encrypted += labels.size(0)

encrypted_accuracy = 100 * correct_encrypted / total_encrypted
print(f"Accuracy of the encrypted model: {encrypted_accuracy:.2f}%")

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

# Function to print confusion matrix
def print_confusion_matrix(conf_matrix, title):
    print(f"\n{title}")
    print("True Label ->")
    print("Predicted Label ↓")
    print(f"{' ':>4}", end="")
    for i in range(10):
        print(f"{i:>4}", end="")
    print()
    print("-" * 44)

    for i, row in enumerate(conf_matrix):
        print(f"{i:>2} |", end="")
        for value in row:
            print(f"{value:>4}", end="")
        print()

# Collect predictions and true labels for the unencrypted model
true_labels = []
predicted_labels_unencrypted = []

model.eval()  # Ensure the model is in evaluation mode
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.numpy())
        predicted_labels_unencrypted.extend(predicted.numpy())

# Confusion matrix for unencrypted model
conf_matrix_unencrypted = confusion_matrix(true_labels, predicted_labels_unencrypted)
print_confusion_matrix(conf_matrix_unencrypted, title="Confusion Matrix for Unencrypted Model")

# Collect predictions and true labels for the encrypted model
predicted_labels_encrypted = []

for images, labels in test_loader:
    preprocessed_images = preprocess_data(images)
    encrypted_outputs = quantized_model.forward(preprocessed_images)
    predicted_encrypted = np.argmax(encrypted_outputs, axis=1)
    predicted_labels_encrypted.extend(predicted_encrypted)

# Confusion matrix for encrypted model
conf_matrix_encrypted = confusion_matrix(true_labels, predicted_labels_encrypted)
print_confusion_matrix(conf_matrix_encrypted, title="Confusion Matrix for Encrypted Model")

