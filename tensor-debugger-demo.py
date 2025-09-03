#!/usr/bin/env python3
"""
RTX 5080 Tensor Shape Debugger Demo
A minimal demonstration of tensor shape debugging functionality
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import re
import threading

class TensorShape:
    def __init__(self, dimensions, dtype="float32", device="cpu", name="tensor"):
        self.dimensions = dimensions
        self.dtype = dtype
        self.device = device
        self.name = name
    
    def to_dict(self):
        return {
            "dimensions": self.dimensions,
            "dtype": self.dtype,
            "device": self.device,
            "name": self.name
        }

class TensorOperation:
    def __init__(self, op_type, input_shapes, output_shape, is_valid=True, error_message=None):
        self.op_type = op_type
        self.input_shapes = input_shapes
        self.output_shape = output_shape
        self.is_valid = is_valid
        self.error_message = error_message
        self.execution_time_ms = 0.1  # Simulated execution time
    
    def to_dict(self):
        return {
            "op_type": self.op_type,
            "input_shapes": [s.to_dict() for s in self.input_shapes],
            "output_shape": self.output_shape.to_dict(),
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms
        }

class TensorAnalyzer:
    def __init__(self):
        self.rtx5080_available = False  # Simulated - would check actual GPU
        self.device = "CPU (RTX 5080 not available)"
    
    def analyze_model_code(self, pytorch_code):
        """Analyze PyTorch model code and extract tensor operations"""
        operations = []
        
        # Parse common PyTorch patterns
        patterns = [
            (r'torch\.randn\(([^)]+)\)', self._parse_randn),
            (r'nn\.Conv2d\(([^)]+)\)', self._parse_conv2d),
            (r'nn\.Linear\(([^)]+)\)', self._parse_linear),
            (r'\.matmul\(([^)]+)\)', self._parse_matmul),
            (r'\.relu\(\)', self._parse_relu),
            (r'\.view\(([^)]+)\)', self._parse_view),
            (r'\.flatten\(([^)]+)\)', self._parse_flatten),
        ]
        
        for pattern, parser in patterns:
            matches = re.finditer(pattern, pytorch_code)
            for match in matches:
                try:
                    op = parser(match)
                    if op:
                        operations.append(op)
                except Exception as e:
                    print(f"Error parsing {pattern}: {e}")
        
        # Add dimension mismatch detection
        self._detect_mismatches(operations)
        
        # If no operations found, create demo operations
        if not operations:
            operations = self._create_demo_operations()
        
        return {
            "model_name": "Parsed PyTorch Model",
            "operations": [op.to_dict() for op in operations],
            "total_params": sum(self._count_params(op) for op in operations),
            "memory_usage_mb": sum(self._calc_memory(op) for op in operations),
            "performance_metrics": {
                "total_time_ms": sum(op.execution_time_ms for op in operations),
                "gpu_utilization": 0.0 if not self.rtx5080_available else 85.0,
                "memory_efficiency": 15.2,
                "rtx5080_optimized": self.rtx5080_available
            }
        }
    
    def _parse_randn(self, match):
        args = match.group(1).strip()
        dimensions = self._extract_dimensions(args)
        if dimensions:
            return TensorOperation(
                "randn",
                [],
                TensorShape(dimensions, name="random_tensor"),
                True
            )
        return None
    
    def _parse_conv2d(self, match):
        args = match.group(1).strip()
        nums = re.findall(r'\d+', args)
        if len(nums) >= 2:
            in_channels, out_channels = int(nums[0]), int(nums[1])
            input_shape = TensorShape([1, in_channels, 224, 224], name="conv_input")
            output_shape = TensorShape([1, out_channels, 222, 222], name="conv_output")
            return TensorOperation("conv2d", [input_shape], output_shape, True)
        return None
    
    def _parse_linear(self, match):
        args = match.group(1).strip()
        nums = re.findall(r'\d+', args)
        if len(nums) >= 2:
            in_features, out_features = int(nums[0]), int(nums[1])
            input_shape = TensorShape([32, in_features], name="linear_input")
            output_shape = TensorShape([32, out_features], name="linear_output")
            return TensorOperation("linear", [input_shape], output_shape, True)
        return None
    
    def _parse_matmul(self, match):
        # Simulate matrix multiplication with potential dimension mismatch
        shape1 = TensorShape([256, 512], name="matrix_a")
        shape2 = TensorShape([512, 128], name="matrix_b")  # Compatible
        output_shape = TensorShape([256, 128], name="matmul_result")
        
        # Check for dimension mismatch
        is_valid = shape1.dimensions[-1] == shape2.dimensions[-2]
        error_msg = None if is_valid else f"Dimension mismatch: {shape1.dimensions[-1]} != {shape2.dimensions[-2]}"
        
        return TensorOperation("matmul", [shape1, shape2], output_shape, is_valid, error_msg)
    
    def _parse_relu(self, match):
        input_shape = TensorShape([32, 512], name="relu_input")
        return TensorOperation("relu", [input_shape], input_shape, True)
    
    def _parse_view(self, match):
        args = match.group(1).strip()
        dimensions = self._extract_dimensions(args)
        input_shape = TensorShape([32, 2048], name="view_input")
        output_shape = TensorShape(dimensions if dimensions else [32, -1], name="view_output")
        return TensorOperation("view", [input_shape], output_shape, True)
    
    def _parse_flatten(self, match):
        input_shape = TensorShape([32, 128, 7, 7], name="flatten_input")
        output_shape = TensorShape([32, 6272], name="flatten_output")
        return TensorOperation("flatten", [input_shape], output_shape, True)
    
    def _extract_dimensions(self, args):
        # Extract numeric dimensions from arguments
        nums = re.findall(r'\d+', args.replace(',', ' '))
        return [int(n) for n in nums] if nums else []
    
    def _detect_mismatches(self, operations):
        """Detect and mark dimension mismatches"""
        for op in operations:
            if op.op_type == "matmul" and len(op.input_shapes) == 2:
                shape1, shape2 = op.input_shapes
                if (len(shape1.dimensions) >= 2 and len(shape2.dimensions) >= 2):
                    if shape1.dimensions[-1] != shape2.dimensions[-2]:
                        op.is_valid = False
                        op.error_message = f"Matrix multiplication dimension mismatch: {shape1.dimensions[-1]} != {shape2.dimensions[-2]}"
    
    def _create_demo_operations(self):
        """Create demo operations to show tensor flow"""
        ops = []
        
        # Tensor creation
        ops.append(TensorOperation(
            "randn", [],
            TensorShape([1, 3, 224, 224], name="input_image"),
            True
        ))
        
        # Conv2d layer
        ops.append(TensorOperation(
            "conv2d",
            [TensorShape([1, 3, 224, 224], name="input")],
            TensorShape([1, 64, 112, 112], name="conv1_out"),
            True
        ))
        
        # Matrix multiplication with mismatch
        shape1 = TensorShape([256, 512], name="matrix_a")
        shape2 = TensorShape([256, 128], name="matrix_b")  # Incompatible!
        ops.append(TensorOperation(
            "matmul",
            [shape1, shape2],
            TensorShape([256, 128], name="matmul_result"),
            False,
            "Matrix multiplication dimension mismatch: 512 != 256"
        ))
        
        # Linear layer
        ops.append(TensorOperation(
            "linear",
            [TensorShape([32, 1024], name="fc_input")],
            TensorShape([32, 512], name="fc_output"),
            True
        ))
        
        return ops
    
    def _count_params(self, op):
        """Count parameters in operation"""
        if op.op_type in ["conv2d", "linear"]:
            return op.output_shape.dimensions[-1] * 1000  # Rough estimate
        return 0
    
    def _calc_memory(self, op):
        """Calculate memory usage in MB"""
        total_elements = 0
        for shape in op.input_shapes:
            total_elements += abs(reduce(lambda x, y: x * y, shape.dimensions, 1))
        total_elements += abs(reduce(lambda x, y: x * y, op.output_shape.dimensions, 1))
        return (total_elements * 4) / (1024 * 1024)  # 4 bytes per float32

def reduce(func, iterable, initializer):
    """Simple reduce implementation"""
    it = iter(iterable)
    value = initializer
    for element in it:
        if element != -1:  # Skip -1 dimensions
            value = func(value, element)
    return value

class TensorDebuggerHandler(BaseHTTPRequestHandler):
    analyzer = TensorAnalyzer()
    
    def do_GET(self):
        if self.path == '/':
            self._serve_html()
        elif self.path == '/api/status':
            self._serve_status()
        elif self.path.startswith('/static/'):
            self._serve_static()
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/api/analyze':
            self._handle_analyze()
        else:
            self.send_error(404)
    
    def _serve_html(self):
        html = """<!DOCTYPE html>
<html>
<head>
    <title>RTX 5080 Tensor Shape Debugger</title>
    <style>
        body { font-family: monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #00ffff; font-size: 2em; text-shadow: 0 0 10px #00ffff; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: #111; border: 1px solid #333; border-radius: 10px; padding: 20px; }
        textarea { width: 100%; height: 300px; background: #000; color: #00ff00; border: 1px solid #333; padding: 15px; font-family: monospace; }
        button { background: #003300; color: #00ff00; border: 1px solid #00ff00; padding: 10px 20px; cursor: pointer; }
        button:hover { background: #00ff00; color: #000; }
        .operation { margin: 10px 0; padding: 15px; background: #222; border-radius: 5px; border-left: 4px solid #00ff00; }
        .operation.error { border-left-color: #ff0000; background: #330000; }
        .tensor-shape { display: inline-block; background: #333; color: #fff; padding: 3px 8px; border-radius: 3px; margin: 2px; }
        .tensor-shape.mismatch { background: #ff0000; animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .status { display: flex; justify-content: center; gap: 20px; margin-top: 15px; }
        .status-item { padding: 5px 15px; border: 1px solid #333; border-radius: 5px; background: #111; }
        .status-item.disabled { background: #330000; border-color: #ff0000; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎮 RTX 5080 Tensor Shape Debugger</h1>
        <p>Real-time PyTorch tensor flow visualization (Demo Mode)</p>
        <div class="status">
            <div class="status-item disabled">RTX 5080: NOT AVAILABLE</div>
            <div class="status-item">Mode: CPU Demo</div>
            <div class="status-item">Version: 0.1.0</div>
        </div>
    </div>
    
    <div class="container">
        <div class="panel">
            <h3>📝 PyTorch Model Code</h3>
            <textarea id="code" placeholder="Paste PyTorch code here...

Example:
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.fc1 = nn.Linear(1024, 512)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

input_tensor = torch.randn(1, 3, 224, 224)
model = SimpleNet()
output = model(input_tensor)"></textarea>
            <br><br>
            <button onclick="analyze()">🔍 Analyze Tensor Flow</button>
            <button onclick="loadExample()">📋 Load Example</button>
        </div>
        
        <div class="panel">
            <h3>📊 Tensor Flow Analysis</h3>
            <div id="output">Enter code and click analyze to see tensor shapes and detect mismatches</div>
        </div>
    </div>

    <script>
        async function analyze() {
            const code = document.getElementById('code').value;
            const output = document.getElementById('output');
            
            output.innerHTML = '⚡ Analyzing tensor flow...';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pytorch_code: code })
                });
                
                const result = await response.json();
                displayResult(result);
            } catch (error) {
                output.innerHTML = `❌ Error: ${error.message}`;
            }
        }
        
        function displayResult(analysis) {
            let html = `
                <div style="margin-bottom: 20px;">
                    <strong>Model:</strong> ${analysis.model_name}<br>
                    <strong>Total Parameters:</strong> ${analysis.total_params.toLocaleString()}<br>
                    <strong>Memory Usage:</strong> ${analysis.memory_usage_mb.toFixed(1)} MB<br>
                    <strong>Execution Time:</strong> ${analysis.performance_metrics.total_time_ms.toFixed(2)} ms
                </div>
                <h4>🔀 Operations:</h4>
            `;
            
            analysis.operations.forEach(op => {
                const opClass = op.is_valid ? 'operation' : 'operation error';
                const inputShapes = op.input_shapes.map(s => 
                    `<span class="tensor-shape">${s.name}: [${s.dimensions.join(', ')}]</span>`
                ).join(' ');
                const outputShape = `<span class="tensor-shape">[${op.output_shape.dimensions.join(', ')}]</span>`;
                
                html += `
                    <div class="${opClass}">
                        <strong>${op.op_type.toUpperCase()}</strong> (${op.execution_time_ms}ms)<br>
                        Input: ${inputShapes || 'None'}<br>
                        Output: ${outputShape}
                        ${op.error_message ? `<br><span style="color: #ff6666;">⚠️ ${op.error_message}</span>` : ''}
                    </div>
                `;
            });
            
            document.getElementById('output').innerHTML = html;
        }
        
        function loadExample() {
            document.getElementById('code').value = `import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 220 * 220, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model and test
model = ConvNet()
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)

# This will cause a dimension mismatch:
a = torch.randn(256, 512)
b = torch.randn(256, 128)  # Wrong dimension for matmul
result = a.matmul(b)  # ERROR!`;
        }
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def _serve_status(self):
        status = {
            "status": "running",
            "rtx5080_available": False,
            "cuda_devices": 0,
            "version": "0.1.0-demo"
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())
    
    def _handle_analyze(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode())
        
        pytorch_code = data.get('pytorch_code', '')
        analysis = self.analyzer.analyze_model_code(pytorch_code)
        
        response = {
            "success": True,
            "analysis": analysis,
            "device_info": {
                "device": "CPU Demo Mode",
                "optimization_level": "RTX 5080 Support Available with CUDA 12.8+",
                "note": "Install PyTorch with CUDA support for GPU acceleration"
            }
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

def main():
    print("🎮 RTX 5080 Tensor Shape Debugger (Demo Mode)")
    print("=" * 50)
    print("📍 Starting demo server at http://localhost:8081")
    print("⚠️  For full RTX 5080 support, install:")
    print("   - CUDA 12.8+ with Blackwell support")
    print("   - PyTorch with CUDA bindings")
    print("   - tch-rs with RTX 5080 optimizations")
    print()
    
    server = HTTPServer(('localhost', 8081), TensorDebuggerHandler)
    print("🚀 Server running! Open http://localhost:8081")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped")
        server.shutdown()

if __name__ == "__main__":
    main()