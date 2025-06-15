#include <hip/hip_runtime.h>

__global__ void add(float* A, float* B, float* C, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N){
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

void launch_add(float* A, float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    add<<<blocks, threads>>>(A, B, C, M, N);
}

__global__ void sub(float* A, float* B, float* C, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N){
        int idx = row * N + col;
        C[idx] = A[idx] - B[idx];
    }
}   

void launch_sub(float* A, float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    sub<<<blocks, threads>>>(A, B, C, M, N);
}

__global__ void mul(float* A, float* B, float C, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N){
        int idx = row * N + col;
        B[idx] = A[idx] * C;
    }
}
    