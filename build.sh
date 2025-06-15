#!/bin/bash
HIPCC=hipcc
INCLUDES="-Isrc/headers -Isrc/tensor -Isrc/grad"
# Kernel files
KERNELS="src/kernels/matrix.hip src/kernels/attention.hip src/kernels/activations.hip src/kernels/tiled_matrix.hip src/kernels/softmax.hip"
# Gradient files
GRADIENTS="src/grad/backward.hip"
# Model files
MODELS="src/models/transformer.hip"
# Tensor implementation
TENSOR="src/tensor/tensor.hip"
# Main/test files
MAIN="src/main.hip"
# All sources
SOURCES="$KERNELS $GRADIENTS $MODELS $TENSOR $MAIN"
# Output
OUTPUT="bin/test_grad"
# Build
mkdir -p bin
$HIPCC $INCLUDES $SOURCES -o $OUTPUT