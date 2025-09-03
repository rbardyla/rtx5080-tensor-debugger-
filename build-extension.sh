#!/bin/bash

# RTX 5080 Tensor Debugger VS Code Extension Build Script
# Usage: ./build-extension.sh

echo "🎮 Building RTX 5080 Tensor Debugger VS Code Extension..."

# Check if vsce is installed
if ! command -v vsce &> /dev/null; then
    echo "❌ vsce (Visual Studio Code Extension CLI) is not installed"
    echo "📦 Installing vsce globally..."
    npm install -g vsce
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -f *.vsix

# Install dependencies (if needed)
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Package the extension
echo "📦 Packaging extension..."
vsce package --out rtx5080-tensor-debugger-1.0.0.vsix

if [ $? -eq 0 ]; then
    echo "✅ Extension packaged successfully!"
    echo "📁 Output: rtx5080-tensor-debugger-1.0.0.vsix"
    echo ""
    echo "🚀 To install:"
    echo "   code --install-extension rtx5080-tensor-debugger-1.0.0.vsix"
    echo ""
    echo "🧪 To test:"
    echo "   1. Open VS Code"
    echo "   2. Press F5 to launch extension development host"
    echo "   3. Open test-example.py"
    echo "   4. See red squiggles under dimension mismatches!"
else
    echo "❌ Extension packaging failed"
    exit 1
fi