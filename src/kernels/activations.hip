#include <hip/hip_runtime.h>


__global__ void add_residual(float* x, float* residual, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = x[idx] + residual[idx];
    }
}

void launch_add_residual(float* x, float* residual, float* output, int size) {
    dim3 threads(256);
    dim3 blocks((size + 255) / 256);
    add_residual<<<blocks, threads>>>(x, residual, output, size);
}

__global__ void relu(float* x, float* output, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        output[idx] = fmaxf(0.0f, x[idx]);
    }
}
void launch_relu(float* x, float* output, int size) {
    dim3 threads(256);
    dim3 blocks((size + 255) / 256);
    relu<<<blocks, threads>>>(x, output, size);
}
__global__ void gelu(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    }
}

void launch_gelu(float* input, float* output, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu<<<blocks, threads>>>(input, output, n);
}