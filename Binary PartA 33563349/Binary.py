# Imports necessary libraries/modules
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import os
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# Set CUDA_LAUNCH_BLOCKING to 1 for better error messages with CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Read the CSV file
data = pd.read_csv('./data/TrainingDataBinary.csv', header=None)

# Extract features and labels from the data
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)

# Converts the training and testing data to PyTorch tensors
train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

# Print the training and testing tensors
print(train_features)
print(train_labels)
print(test_features)
print(test_labels)
import torch

# Count the number of samples for each label in the training and testing sets
test_label_counts = torch.unique(test_labels, return_counts=True)
train_label_counts = torch.unique(train_labels, return_counts=True)

print("Test Label Counts:")
for label, count in zip(test_label_counts[0], test_label_counts[1]):
    print(f"Label {label}: {count}")

print("\nTrain Label Counts:")
for label, count in zip(train_label_counts[0], train_label_counts[1]):
    print(f"Label {label}: {count}")

# Define the MLPClassifier model using PyTorch's nn.Module
class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

learning_rate = 0.0001
num_epochs = 7500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = train_features.shape[1]
num_classes = len(torch.unique(train_labels))

# Create an instance of the MLPClassifier model
model = MLPClassifier(input_size, num_classes)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pretrained_weights_file = 'model_weights_bin.pt'
pretrained_weights_exist = False
pretrained_weights_exist = os.path.isfile(pretrained_weights_file)

# Check if pretrained weights file exists, and load weights into the model if it does
if pretrained_weights_exist:
    model.load_state_dict(torch.load(pretrained_weights_file, map_location=device))
    pretrained_weights_exist = True

# Train the model if pretrained weights file doesn't exist
if not pretrained_weights_exist:

    # Move the data to the specified device
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    # Train the model
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(train_features)
        loss = criterion(outputs, train_labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    # Save the trained model weights
    torch.save(model.state_dict(), pretrained_weights_file)


test_features = test_features.to(device)
test_labels = test_labels.to(device)

# Evaluate the model on the testing data
with torch.no_grad():
    model.eval()
    test_outputs = model(test_features)
    _, predicted_labels = torch.max(test_outputs, 1)

    # Convert tensors to numpy arrays
    predicted_labels = predicted_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    # Calculate classification metrics
    accuracy = accuracy_score(test_labels, predicted_labels)
    precision = precision_score(test_labels, predicted_labels, average='weighted')
    recall = recall_score(test_labels, predicted_labels, average='weighted')
    f1 = f1_score(test_labels, predicted_labels, average='weighted')

    # Print the classification metrics
    print("Classification Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Calculate overall accuracy
    correct = (predicted_labels == test_labels).sum().item()
    total = test_labels.size
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")

    # Output the Confusion Matrix
    y_true = test_labels
    y_pred = predicted_labels
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["0", "1"], cmap=plt.cm.Reds, colorbar=True)
    plt.title("Confusion Matrix")
    plt.show()

# Read the testing data from a CSV file
data = pd.read_csv('./data/TestingDataBinary.csv', header=None)
test_features = torch.tensor(data.values, dtype=torch.float32)

# Create a new instance of the MLPClassifier model
model = MLPClassifier(input_size, num_classes)

# Load the pretrained model weights
model.load_state_dict(torch.load('model_weights_bin.pt', map_location=torch.device('cpu')))
model = model.to(device)

test_features = test_features.to(device)

# Perform inference on the testing data
with torch.no_grad():
    model.eval()
    test_outputs = model(test_features)
    _, predicted_labels = torch.max(test_outputs, 1)

# Convert predicted labels to numpy array
predicted_labels = predicted_labels.cpu().numpy()

# Show all predicted labels
print(predicted_labels)

# Add the predicted labels to the original testing data DataFrame
data['Predicted Label'] = predicted_labels

# Save the DataFrame to a CSV file
data.to_csv('TestingResultsBinary.csv', index=False)

