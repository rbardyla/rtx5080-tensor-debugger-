"""
Test file for RTX 5080 Tensor Debugger VS Code Extension
This file contains intentional tensor dimension bugs for testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BuggyNet(nn.Module):
    """A neural network with intentional tensor dimension bugs"""
    
    def __init__(self):
        super(BuggyNet, self).__init__()
        
        # Linear layers with dimension mismatches
        self.fc1 = nn.Linear(784, 128)          # ✅ Correct
        self.fc2 = nn.Linear(256, 64)           # ❌ Should be 128, not 256
        self.fc3 = nn.Linear(64, 32)            # ✅ Correct
        self.fc4 = nn.Linear(16, 10)            # ❌ Should be 32, not 16
        
        # Conv2d layers with channel mismatches
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)    # ✅ Correct (RGB input)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)  # ❌ Should be 32, not 64
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1) # ✅ Correct
        
        # More complex cases
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Forward pass that would fail due to dimension mismatches
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # This will fail: expected 128, got 256
        x = F.relu(self.fc3(x))
        x = self.fc4(x)          # This will fail: expected 32, got 16
        return x

class ConvNet(nn.Module):
    """Convolutional network with channel mismatches"""
    
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3)       # ✅ RGB -> 32 channels
        self.conv2 = nn.Conv2d(16, 64, 3)      # ❌ Should be 32, not 16
        self.conv3 = nn.Conv2d(64, 128, 3)     # ✅ Correct
        self.conv4 = nn.Conv2d(256, 512, 3)    # ❌ Should be 128, not 256
        
        # This would work if dimensions were correct
        self.fc = nn.Linear(512 * 4 * 4, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # Channel mismatch here
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # Another mismatch
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CorrectNet(nn.Module):
    """A network with correct tensor dimensions - should have no issues"""
    
    def __init__(self):
        super(CorrectNet, self).__init__()
        
        # All dimensions match correctly
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        
        # Conv layers with correct channels
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Test instantiation
if __name__ == "__main__":
    # These should trigger dimension errors when analyzed
    buggy_net = BuggyNet()
    conv_net = ConvNet()
    
    # This should be clean
    correct_net = CorrectNet()
    
    print("RTX 5080 Tensor Debugger should detect dimension mismatches in BuggyNet and ConvNet!")
    print("CorrectNet should have no issues detected.")