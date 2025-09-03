# Contributing to RTX 5080 Tensor Debugger

Thanks for your interest in improving the tensor debugger! 🎮

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**: `git clone https://github.com/yourusername/rtx5080-tensor-debugger.git`
3. **Test the current version**: `python3 test_tensor_debugger.py`
4. **Make your changes**
5. **Test your changes**: `python3 tensor-debugger-demo.py`
6. **Submit a pull request**

## 🎯 Areas Where We Need Help

### High Priority
- **🔍 More PyTorch Operations**: Add support for Conv2D, LSTM, Attention, etc.
- **🐛 Edge Case Handling**: Handle malformed PyTorch code gracefully  
- **⚡ Performance**: Optimize analysis for larger models (1000+ layers)
- **🌐 Web Interface**: Improve the HTML/CSS for better UX

### Medium Priority  
- **📊 Better Visualizations**: More intuitive tensor shape diagrams
- **🔧 Auto-Fix Code**: Generate corrected PyTorch code automatically
- **📱 Sharing Features**: Better shareable result links
- **🎮 RTX 5080 Integration**: Full CUDA implementation

### Low Priority
- **📚 Documentation**: More examples and tutorials
- **🧪 Test Coverage**: Unit tests for edge cases
- **🎨 UI Polish**: Better colors, animations, etc.

## 🛠 Development Setup

### Basic Development (Demo Version)
```bash
# No dependencies needed! Uses Python standard library only
python3 tensor-debugger-demo.py
```

### Full RTX 5080 Development  
```bash
# Requires CUDA 12.8+ and RTX 5080
pip install torch tch numpy
cargo build --release
```

## 🧪 Testing Your Changes

### Run All Tests
```bash
python3 test_tensor_debugger.py
python3 examples/common_bugs.py
```

### Manual Testing
```bash
# Start the demo server
python3 tensor-debugger-demo.py

# Test with buggy code in browser at localhost:8082
```

## 📝 Code Style

- **Keep it simple**: This is a minimal tool, not a framework
- **No external dependencies** for demo version
- **Clear variable names**: `input_dimensions` not `inp_dim`  
- **Helpful error messages**: Users should understand what went wrong
- **Performance matters**: ML engineers are impatient

## 🎯 Bug Report Guidelines

When reporting bugs, please include:
1. **PyTorch code** that caused the issue
2. **Expected behavior** vs **actual behavior**  
3. **Error messages** (if any)
4. **System info**: Python version, OS, GPU (if applicable)

## 🚀 Feature Request Guidelines

We prioritize features that:
1. **Save debugging time** for ML engineers
2. **Are simple to use** (one-click, copy-paste, etc.)
3. **Work immediately** (no complex setup)
4. **Solve common problems** (affect many users)

## ❌ What We Won't Accept

- **Complex dependencies**: Keep the demo dependency-free
- **Unrelated features**: Stay focused on tensor debugging
- **Breaking changes**: Don't break existing functionality  
- **Performance regressions**: Don't make analysis slower

## 🎉 Recognition

Contributors will be:
- **Listed in README** with GitHub profile links
- **Tagged in releases** that include their contributions  
- **Given credit** in any academic papers or blog posts
- **Invited to collaborate** on future ML tooling projects

## 🤝 Questions?

- **Open an issue** for bugs or feature requests
- **Start a discussion** for general questions
- **Email directly** for sensitive topics

Thanks for helping make PyTorch debugging faster for everyone! 🚀