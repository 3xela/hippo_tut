#include <hip/hip_runtime.h>


__global__ void matmul_backward_A(float* grad_C, float* B, float* grad_A, int M, int N, int K) {
    // grad_A = grad_C @ B^T
    // Dimensions: (M,N) @ (N,K)^T = (M,N) @ (K,N) = (M,K)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            sum += grad_C[row * N + n] * B[col * N + n];  // B transposed
        }
        atomicAdd(&grad_A[row * K + col], sum);  // Accumulate gradients
    }
}

__global__ void matmul_backward_B(float* A, float* grad_C, float* grad_B, int M, int N, int K) {
    // grad_B = A^T @ grad_C  
    // Dimensions: (M,K)^T @ (M,N) = (K,M) @ (M,N) = (K,N)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < K && col < N) {
        float sum = 0.0f;
        for (int m = 0; m < M; m++) {
            sum += A[m * K + row] * grad_C[m * N + col];  // A transposed
        }
        atomicAdd(&grad_B[row * N + col], sum);  // Accumulate gradients
    }
}

void launch_matmul_backward_A(float* grad_C, float* B, float* grad_A, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((K + 15) / 16, (M + 15) / 16);
    matmul_backward_A<<<blocks, threads>>>(grad_C, B, grad_A, M, N, K);
}

void launch_matmul_backward_B(float* A, float* grad_C, float* grad_B, int M, int N, int K) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (K + 15) / 16);
    matmul_backward_B<<<blocks, threads>>>(A, grad_C, grad_B, M, N, K);
}