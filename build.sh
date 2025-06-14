#!/bin/bash
HIPCC=hipcc
INCLUDES="-Isrc/headers -Isrc/tensor"
# Kernel files
KERNELS="src/kernels/matrix.hip src/kernels/attention.hip src/kernels/activations.hip src/kernels/tiled_matrix.hip src/kernels/softmax.hip"
# Model files
MODELS="src/models/transformer.hip"
# Tensor implementation
TENSOR="src/tensor/tensor.hip"
# Main/test files
MAIN="src/main.hip"
# All sources
SOURCES="$KERNELS $MODELS $TENSOR $MAIN"
# Output
OUTPUT="bin/transformer"
# Build
mkdir -p bin
$HIPCC $INCLUDES $SOURCES -o $OUTPUT