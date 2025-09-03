#!/usr/bin/env python3
"""
CRITICAL VALIDATION: Test if our tensor debugger actually works
Compare our analysis vs real PyTorch runtime errors
"""

import torch
import torch.nn as nn
import sys
import traceback

# Test cases with KNOWN bugs that should crash PyTorch
test_cases = [
    {
        "name": "Dimension Mismatch (Should Crash)",
        "code": """
class BuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(256, 64)  # BUG: Should be 128, not 256
        self.layer3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)  # This should crash
        x = self.layer3(x)
        return x

model = BuggyNet()
input_tensor = torch.randn(32, 784)
output = model(input_tensor)
""",
        "expected_error": "Linear layer input mismatch",
        "our_prediction": "Layer 1 outputs 128, but Layer 2 expects 256"
    },
    
    {
        "name": "Working Model (Should NOT Crash)",
        "code": """
class WorkingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)  # Correct: matches layer1 output
        self.layer3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = WorkingNet()
input_tensor = torch.randn(32, 784)
output = model(input_tensor)
""",
        "expected_error": None,
        "our_prediction": "No dimension errors"
    },
    
    {
        "name": "Complex ResNet Bug (Should Crash)",
        "code": """
class BuggyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(3072, 512)  # 32*32*3 = 3072
        self.layer2 = nn.Linear(256, 256)   # BUG: Should be 512, not 256
        self.layer3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.flatten(x)  # [batch, 3, 32, 32] -> [batch, 3072]
        x = self.layer1(x)   # [batch, 3072] -> [batch, 512]
        x = self.layer2(x)   # [batch, 512] -> CRASH! Expected [batch, 256]
        x = self.layer3(x)
        return x

model = BuggyResNet()
input_tensor = torch.randn(32, 3, 32, 32)
output = model(input_tensor)
""",
        "expected_error": "Linear layer input mismatch", 
        "our_prediction": "Layer 1 outputs 512, but Layer 2 expects 256"
    }
]

def test_pytorch_runtime(code_str):
    """Run PyTorch code and capture what actually happens"""
    try:
        # Execute the PyTorch code
        exec(code_str)
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

def test_our_analyzer(code_str):
    """Test our tensor shape analyzer"""
    import re
    
    # Our simple pattern matching (what we actually built)
    linear_pattern = r'nn\.Linear\((\d+),\s*(\d+)\)'
    layers = re.findall(linear_pattern, code_str)
    
    errors = []
    for i in range(len(layers) - 1):
        current_out = int(layers[i][1])
        next_in = int(layers[i+1][0])
        
        if current_out != next_in:
            errors.append(f"Layer {i+1} outputs {current_out}, but Layer {i+2} expects {next_in}")
    
    return {"errors": errors, "layers_found": len(layers)}

def main():
    print("🧪 CRITICAL VALIDATION: Does Our Tool Actually Work?")
    print("=" * 70)
    
    all_correct = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 TEST {i}: {test_case['name']}")
        print("-" * 50)
        
        # Test our analyzer
        our_result = test_our_analyzer(test_case['code'])
        print(f"🔍 Our Analysis: {our_result}")
        
        # Test real PyTorch
        pytorch_result = test_pytorch_runtime(test_case['code'])
        print(f"🎮 PyTorch Runtime: {pytorch_result['success']}")
        if not pytorch_result['success']:
            print(f"   Error: {pytorch_result['error'][:100]}...")
        
        # Validate our prediction
        if test_case['expected_error'] is None:
            # Should work
            if pytorch_result['success'] and len(our_result['errors']) == 0:
                print("✅ CORRECT: We predicted no errors, PyTorch ran successfully")
            else:
                print("❌ WRONG: Mismatch between our prediction and PyTorch reality")
                all_correct = False
        else:
            # Should crash
            if not pytorch_result['success'] and len(our_result['errors']) > 0:
                print("✅ CORRECT: We predicted errors, PyTorch crashed")
            else:
                print("❌ WRONG: Mismatch between our prediction and PyTorch reality")
                all_correct = False
                
        print(f"   Expected: {test_case['expected_error']}")
        print(f"   Our Prediction: {test_case['our_prediction']}")
    
    print("\n" + "=" * 70)
    if all_correct:
        print("🎉 VALIDATION PASSED: Our tool correctly predicts PyTorch behavior!")
        print("✅ Safe to show to users")
    else:
        print("🚨 VALIDATION FAILED: Our tool gives wrong answers!")
        print("❌ DO NOT show to users until fixed")
        print("\n🔧 Our tool needs debugging before we can debug other people's code!")
    
    return all_correct

if __name__ == "__main__":
    validation_passed = main()
    sys.exit(0 if validation_passed else 1)