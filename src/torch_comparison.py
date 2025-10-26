import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the input layer
input_data = np.array([
    [1.0, 0.5, 1.2, 0.8, 1.5, 0.9, 1.3, 0.7, 1.1],
    [0.6, 1.4, 0.8, 1.7, 1.0, 1.6, 0.9, 1.2, 0.5],
    [1.3, 0.7, 1.8, 1.1, 1.4, 0.6, 1.9, 1.0, 1.5],
    [0.9, 1.2, 0.8, 1.6, 1.3, 1.1, 0.7, 1.4, 0.9],
    [1.1, 0.8, 1.5, 1.0, 1.7, 1.2, 1.4, 0.6, 1.3],
    [0.7, 1.3, 0.9, 1.4, 1.1, 1.8, 1.0, 1.5, 0.8],
    [1.2, 0.6, 1.4, 0.9, 1.3, 1.0, 1.6, 1.1, 1.7],
    [0.8, 1.1, 0.7, 1.5, 1.2, 1.4, 0.9, 1.3, 1.0],
    [1.0, 1.5, 0.8, 1.2, 0.9, 1.3, 1.1, 0.7, 1.4]
])

# Convert to tensor and add batch and channel dimensions
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 9, 9)

# Define the target (one-hot encoded)
target = torch.tensor([4], dtype=torch.long)  # Class 4 is the target (index 4 in 0-based indexing)

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # First conv layer with 5x5 weights
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0, bias=False)
        
        # Second conv layer with 3x3 weights  
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        
        # Set the custom weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Set first conv weights
        conv1_weights = torch.tensor([
            [0.1, 0.2, 0.3, 0.2, 0.1],
            [0.2, 0.4, 0.6, 0.4, 0.2],
            [0.3, 0.6, 0.9, 0.6, 0.3],
            [0.2, 0.4, 0.6, 0.4, 0.2],
            [0.1, 0.2, 0.3, 0.2, 0.1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Set second conv weights
        conv2_weights = torch.tensor([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.5],
            [0.2, 0.5, 1.0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.conv1.weight.data = conv1_weights
        self.conv2.weight.data = conv2_weights
        
    def forward(self, x):
        # Print input shape
        print(f"Input shape: {x.shape}")
        
        # First convolution
        x = self.conv1(x)
        print(f"After conv1 shape: {x.shape}")
        print(f"conv1 weights shape: {self.conv1.weight.shape}")
        print(f"conv1 weights:\n{self.conv1.weight.data.squeeze()}")
        
        # Second convolution  
        x = self.conv2(x)
        print(f"After conv2 shape: {x.shape}")
        print(f"conv2 weights shape: {self.conv2.weight.shape}")
        print(f"conv2 weights:\n{self.conv2.weight.data.squeeze()}")
        
        # Flatten for softmax
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
        print(f"Flattened shape: {x.shape}")
        
        return x

# Create model
model = CustomCNN()

# Print initial weights
print("=== INITIAL WEIGHTS ===")
print("Conv1 weights:")
print(model.conv1.weight.data.squeeze())
print("\nConv2 weights:")
print(model.conv2.weight.data.squeeze())

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("\n=== FORWARD PASS ===")
# Forward pass
output = model(input_tensor)
print(f"Output before softmax:\n{output}")

# Apply softmax manually to see probabilities
softmax = nn.Softmax(dim=1)
probs = softmax(output)
print(f"Probabilities after softmax:\n{probs}")

print("\n=== LOSS COMPUTATION ===")
# Compute loss
loss = criterion(output, target)
print(f"Loss: {loss.item()}")

print("\n=== BACKWARD PASS ===")
# Zero gradients
optimizer.zero_grad()

# Backward pass
loss.backward()

# Print gradients
print("Gradients for conv1:")
print(model.conv1.weight.grad.squeeze())
print("\nGradients for conv2:")
print(model.conv2.weight.grad.squeeze())

print("\n=== WEIGHT UPDATE ===")
# Perform optimization step
optimizer.step()

# Print updated weights
print("Updated conv1 weights:")
print(model.conv1.weight.data.squeeze())
print("\nUpdated conv2 weights:")
print(model.conv2.weight.data.squeeze())

# Additional detailed information
print("\n=== DETAILED INFORMATION ===")
print(f"Input shape: {input_tensor.shape}")
print(f"After conv1 (5x5, stride=1, no padding): {(9-5+1, 9-5+1)} = 5x5")
print(f"After conv2 (3x3, stride=1, no padding): {(5-3+1, 5-3+1)} = 3x3")
print(f"Flattened output size: {3*3} = 9 classes")
print(f"Target class index: {target.item()} (5th position in 0-based indexing)")
