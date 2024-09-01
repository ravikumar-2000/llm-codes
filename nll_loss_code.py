import torch
import torch.nn as nn
import torch.nn.functional as F

# Example logits (output from the model)
logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]])

# Step 1: Compute log-softmax of logits
# This step converts logits to log-probabilities
log_softmax_output = F.log_softmax(logits, dim=-1)

print("Log-Softmax Output:")
print(log_softmax_output)

# Output:
# tensor([[-0.4170, -1.4170, -2.3170],
#         [-1.4170, -0.4170, -2.3170]])

# Step 2: Define the target class indices
# These are the correct classes for each sample in the batch
targets = torch.tensor(
    [0, 1]
)  # Correct classes: first sample is class 0, second sample is class 1

print("\nTargets:")
print(targets)

# Output:
# tensor([0, 1])

# Step 3: Define NLLLoss criterion
# This initializes the loss function
criterion = nn.NLLLoss()

# Step 4: Compute the loss
# This calculates the negative log-likelihood loss
loss = criterion(log_softmax_output, targets)

print("\nLoss:")
print(loss.item())

# Output:
# 0.417022705078125
