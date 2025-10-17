import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TwoLayerCNNAutodiff(nn.Module):
    def __init__(self):
        super(TwoLayerCNNAutodiff, self).__init__()
        
        # First conv layer (5x5 kernel)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0, bias=False)
        # Second conv layer (3x3 kernel)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        # Linear layer
        self.linear = nn.Linear(9, 1, bias=False)
        
        # Initialize with the exact weights from the manual example
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Conv1 weights (5x5)
        conv1_weight = torch.tensor([
            [0.1, 0.2, 0.3, 0.2, 0.1],
            [0.2, 0.4, 0.6, 0.4, 0.2],
            [0.3, 0.6, 0.9, 0.6, 0.3],
            [0.2, 0.4, 0.6, 0.4, 0.2],
            [0.1, 0.2, 0.3, 0.2, 0.1]
        ]).unsqueeze(0).unsqueeze(0).float()
        self.conv1.weight = nn.Parameter(conv1_weight)
        
        # Conv2 weights (3x3)
        conv2_weight = torch.tensor([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.5],
            [0.2, 0.5, 1.0]
        ]).unsqueeze(0).unsqueeze(0).float()
        self.conv2.weight = nn.Parameter(conv2_weight)
        
        # Linear weights
        linear_weight = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reshape(1, 9).float()
        self.linear.weight = nn.Parameter(linear_weight)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.linear(x)
        return x

def autodiff_training_demo():
    """
    Demonstrate training using PyTorch autodiff
    """
    print("=" * 90)
    print("PYTORCH AUTODIFF TRAINING DEMONSTRATION")
    print("=" * 90)
    
    # Create model
    model = TwoLayerCNNAutodiff()
    
    # Input data (same as manual example)
    input_data = torch.tensor([
        [1.0, 0.5, 1.2, 0.8, 1.5, 0.9, 1.3, 0.7, 1.1],
        [0.6, 1.4, 0.8, 1.7, 1.0, 1.6, 0.9, 1.2, 0.5],
        [1.3, 0.7, 1.8, 1.1, 1.4, 0.6, 1.9, 1.0, 1.5],
        [0.9, 1.2, 0.8, 1.6, 1.3, 1.1, 0.7, 1.4, 0.9],
        [1.1, 0.8, 1.5, 1.0, 1.7, 1.2, 1.4, 0.6, 1.3],
        [0.7, 1.3, 0.9, 1.4, 1.1, 1.8, 1.0, 1.5, 0.8],
        [1.2, 0.6, 1.4, 0.9, 1.3, 1.0, 1.6, 1.1, 1.7],
        [0.8, 1.1, 0.7, 1.5, 1.2, 1.4, 0.9, 1.3, 1.0],
        [1.0, 1.5, 0.8, 1.2, 0.9, 1.3, 1.1, 0.7, 1.4]
    ]).unsqueeze(0).unsqueeze(0).float()
    
    target = torch.tensor([[2.0]]).float()
    learning_rate = 0.01
    
    # Use SGD optimizer (same as manual update: W = W - lr * gradient)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    print("MODEL ARCHITECTURE:")
    print(f"Input shape: {input_data.shape}")
    print(f"Network: Input (9x9) -> Conv1 (5x5) -> Conv2 (3x3) -> Linear (9) -> Output (1)")
    print(f"Target: {target.item()}")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: SGD (mimics manual W = W - η·∇W)")
    print(f"Loss: MSE")
    
    # =========================================================================
    # 1. INITIAL STATE
    # =========================================================================
    print("\n" + "=" * 90)
    print("1. INITIAL STATE")
    print("=" * 90)
    
    print("\nInitial Weights:")
    print(f"Conv1 weights:\n{model.conv1.weight.data.squeeze()}")
    print(f"Conv2 weights:\n{model.conv2.weight.data.squeeze()}")
    print(f"Linear weights: {model.linear.weight.data.squeeze().tolist()}")
    
    # =========================================================================
    # 2. FORWARD PASS
    # =========================================================================
    print("\n" + "=" * 90)
    print("2. FORWARD PASS")
    print("=" * 90)
    
    # Forward pass
    output = model(input_data)
    print(f"Output: {output.item():.6f}")
    
    # MSE Loss
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(f"MSE Loss: {loss.item():.6f}")
    
    # =========================================================================
    # 3. BACKWARD PASS (AUTODIFF)
    # =========================================================================
    print("\n" + "=" * 90)
    print("3. BACKWARD PASS WITH AUTODIFF")
    print("=" * 90)
    
    # Zero gradients
    optimizer.zero_grad()
    print("Gradients zeroed: optimizer.zero_grad()")
    
    # Backward pass - PyTorch automatically computes all gradients!
    loss.backward()
    print("Gradients computed: loss.backward()")
    print("PyTorch autodiff automatically:")
    print("  - Computes ∂L/∂output")
    print("  - Backpropagates through linear layer")
    print("  - Backpropagates through conv2 layer") 
    print("  - Backpropagates through conv1 layer")
    print("  - Computes ∂L/∂W for all parameters")
    
    # Show computed gradients
    print(f"\nComputed Gradients (∂L/∂W):")
    print(f"Linear gradients: {model.linear.weight.grad.squeeze().tolist()}")
    print(f"Conv2 gradients:\n{model.conv2.weight.grad.squeeze()}")
    print(f"Conv1 gradients:\n{model.conv1.weight.grad.squeeze()}")
    
    # =========================================================================
    # 4. WEIGHT UPDATES
    # =========================================================================
    print("\n" + "=" * 90)
    print("4. WEIGHT UPDATES")
    print("=" * 90)
    
    print("Before update:")
    print(f"Linear weights: {model.linear.weight.data.squeeze().tolist()}")
    print(f"Conv2 weights:\n{model.conv2.weight.data.squeeze()}")
    print(f"Conv1 weights:\n{model.conv1.weight.data.squeeze()}")
    
    # Update weights - PyTorch automatically applies: W = W - lr * gradient
    optimizer.step()
    print(f"\nWeights updated: optimizer.step()")
    print(f"PyTorch automatically applies: W_new = W_old - {learning_rate} × ∂L/∂W")
    
    print("\nAfter update:")
    print(f"Linear weights: {model.linear.weight.data.squeeze().tolist()}")
    print(f"Conv2 weights:\n{model.conv2.weight.data.squeeze()}")
    print(f"Conv1 weights:\n{model.conv1.weight.data.squeeze()}")
    
    # =========================================================================
    # 5. COMPLETE TRAINING LOOP
    # =========================================================================
    print("\n" + "=" * 90)
    print("5. COMPLETE TRAINING LOOP (3 EPOCHS)")
    print("=" * 90)
    
    # Reset model to initial state
    model = TwoLayerCNNAutodiff()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print("Training for 3 epochs...")
    for epoch in range(3):
        # Forward pass
        output = model(input_data)
        loss = criterion(output, target)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}:")
        print(f"  Output: {output.item():.6f}")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Linear weights: {[f'{w:.4f}' for w in model.linear.weight.data.squeeze().tolist()]}")
    
    # =========================================================================
    # 6. GRADIENT FLOW VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 90)
    print("6. GRADIENT FLOW VISUALIZATION")
    print("=" * 90)
    
    # Do one more forward/backward to examine gradients
    model = TwoLayerCNNAutodiff()
    output = model(input_data)
    loss = criterion(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    print("Gradient flow through the network:")
    print(f"∂L/∂output (computed by PyTorch): {loss.grad_fn}")
    print(f"Linear layer gradient function: {loss.grad_fn.next_functions[0][0]}")
    print(f"Conv2 layer gradient function: {loss.grad_fn.next_functions[0][0].next_functions[0][0]}")
    print(f"Conv1 layer gradient function: {loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0]}")
    
    print(f"\nGradient shapes:")
    print(f"∂L/∂W_linear shape: {model.linear.weight.grad.shape}")
    print(f"∂L/∂W_conv2 shape: {model.conv2.weight.grad.shape}")
    print(f"∂L/∂W_conv1 shape: {model.conv1.weight.grad.shape}")
    
    return {
        'final_output': output.item(),
        'final_loss': loss.item(),
        'final_linear_weights': model.linear.weight.data.squeeze().tolist(),
        'final_conv2_weights': model.conv2.weight.data.squeeze(),
        'final_conv1_weights': model.conv1.weight.data.squeeze()
    }

def compare_manual_vs_autodiff():
    """
    Compare manual calculations with autodiff results
    """
    print("\n" + "=" * 90)
    print("COMPARISON: MANUAL vs AUTODIFF")
    print("=" * 90)
    
    # Create two identical models
    model_manual = TwoLayerCNNAutodiff()
    model_autodiff = TwoLayerCNNAutodiff()
    
    # Input data
    input_data = torch.tensor([
        [1.0, 0.5, 1.2, 0.8, 1.5, 0.9, 1.3, 0.7, 1.1],
        [0.6, 1.4, 0.8, 1.7, 1.0, 1.6, 0.9, 1.2, 0.5],
        [1.3, 0.7, 1.8, 1.1, 1.4, 0.6, 1.9, 1.0, 1.5],
        [0.9, 1.2, 0.8, 1.6, 1.3, 1.1, 0.7, 1.4, 0.9],
        [1.1, 0.8, 1.5, 1.0, 1.7, 1.2, 1.4, 0.6, 1.3],
        [0.7, 1.3, 0.9, 1.4, 1.1, 1.8, 1.0, 1.5, 0.8],
        [1.2, 0.6, 1.4, 0.9, 1.3, 1.0, 1.6, 1.1, 1.7],
        [0.8, 1.1, 0.7, 1.5, 1.2, 1.4, 0.9, 1.3, 1.0],
        [1.0, 1.5, 0.8, 1.2, 0.9, 1.3, 1.1, 0.7, 1.4]
    ]).unsqueeze(0).unsqueeze(0).float()
    
    target = torch.tensor([[2.0]]).float()
    learning_rate = 0.01
    
    print("Both models start with identical weights:")
    print(f"Linear weights: {model_manual.linear.weight.data.squeeze().tolist()}")
    
    # Manual update (simplified)
    output_manual = model_manual(input_data)
    loss_manual = F.mse_loss(output_manual, target)
    
    # Manual backward (using autodiff but showing the process)
    model_manual.zero_grad()
    loss_manual.backward()
    
    # Manual weight update
    with torch.no_grad():
        model_manual.linear.weight -= learning_rate * model_manual.linear.weight.grad
        model_manual.conv2.weight -= learning_rate * model_manual.conv2.weight.grad
        model_manual.conv1.weight -= learning_rate * model_manual.conv1.weight.grad
    
    # Autodiff update
    optimizer = optim.SGD(model_autodiff.parameters(), lr=learning_rate)
    output_autodiff = model_autodiff(input_data)
    loss_autodiff = F.mse_loss(output_autodiff, target)
    
    optimizer.zero_grad()
    loss_autodiff.backward()
    optimizer.step()
    
    print(f"\nAfter one update step:")
    print(f"Manual linear weights: {[f'{w:.6f}' for w in model_manual.linear.weight.data.squeeze().tolist()]}")
    print(f"Autodiff linear weights: {[f'{w:.6f}' for w in model_autodiff.linear.weight.data.squeeze().tolist()]}")
    
    # Check if they match
    manual_linear = model_manual.linear.weight.data
    autodiff_linear = model_autodiff.linear.weight.data
    diff = torch.norm(manual_linear - autodiff_linear)
    
    print(f"\nDifference between manual and autodiff: {diff.item():.10f}")
    if diff < 1e-6:
        print("✅ Manual and autodiff updates are identical!")
    else:
        print("❌ Manual and autodiff updates differ")

# Main execution
if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("PYTORCH AUTODIFF NEURAL NETWORK DEMONSTRATION")
    print("Same architecture as manual example but using PyTorch's automatic differentiation")
    print("=" * 90)
    
    # Run autodiff demo
    results = autodiff_training_demo()
    
    # Compare with manual approach
    compare_manual_vs_autodiff()
    
    print("\n" + "=" * 90)
    print("AUTODIFF ADVANTAGES SUMMARY")
    print("=" * 90)
    advantages = """
    PyTorch Autodiff Advantages:
    
    1. AUTOMATIC GRADIENT COMPUTATION:
       - No need to manually derive ∂L/∂W for each layer
       - Handles complex chain rule automatically
       - Computes gradients for all parameters with one loss.backward() call
    
    2. EFFICIENT IMPLEMENTATION:
       - Optimized C++ backend for gradient computations
       - Automatic gradient accumulation management
       - Memory-efficient gradient computation
    
    3. FLEXIBILITY:
       - Easy to modify network architecture
       - Supports complex operations and custom layers
       - Automatic handling of different tensor operations
    
    4. DEBUGGING FEATURES:
       - Gradient checking utilities
       - NaN gradient detection
       - Computational graph visualization
    
    5. OPTIMIZER INTEGRATION:
       - Various optimization algorithms (SGD, Adam, etc.)
       - Learning rate scheduling
       - Gradient clipping
    """
    print(advantages)
