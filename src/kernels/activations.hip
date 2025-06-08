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