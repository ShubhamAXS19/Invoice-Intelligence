#!/bin/bash
set -e

echo "→ Installing PaddlePaddle (CPU, Apple Silicon safe)..."
pip install paddlepaddle==2.6.2 -f https://www.paddlepaddle.org.cn/whl/mac/cpu/stable.html
b
echo "→ Installing PaddleOCR..."
pip install paddleocr==2.7.3

echo "→ Installing remaining dependencies..."
pip install -r requirements.txt --ignore-installed paddlepaddle paddleocr

echo "✓ All dependencies installed."