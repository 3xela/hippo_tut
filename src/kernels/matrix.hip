#include <hip/hip_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void matrix_mul(float* A, float* B, float* C, int M , int N, int K){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N){
        float sum  = 0.0f;
        int idx = row * N + col;
        for (int k = 0; k < K ; k++){
            sum += A[row*K + k] * B[k*N + col];
        }
        C[idx] = sum;
    }
}

__global__ void transpose(float* A, float* B,  int M, int N){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    if (row < M && col < N) {
        tile[threadIdx.y][threadIdx.x] = A[row * N + col];
    }
    __syncthreads();
    
    int transposed_row = blockIdx.x * blockDim.x + threadIdx.y;  // swap x/y
    int transposed_col = blockIdx.y * blockDim.y + threadIdx.x;
    
    if (transposed_row < N && transposed_col < M) {
        B[transposed_row * M + transposed_col] = tile[threadIdx.x][threadIdx.y];
    }

}


__global__ void scale(float* A, float* B, int M , int N, float d ){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N){
        int idx = row * N + col;
        B[idx] = A[idx] * d;
    }

}

void launch_scale(float* A, float* B, int M, int N, float d) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    scale<<<blocks, threads>>>(A, B, M, N, d);
}

void launch_matmul(float* A, float* B, float* C, int M , int N, int K){
    dim3 threads(16,16);
    dim3 blocks((N+15)/16, (M+15)/16);
    matrix_mul<<<blocks,threads>>>(A,B,C,M,N,K);
}

void launch_transpose(float* A, float* B, int M, int N){
    dim3 threads(16,16);
    dim3 blocks((N+15)/16, (M+15)/16);
    transpose<<<blocks,threads>>>(A,B,M,N);
}

// int main(){
//     int N = 1000;
//     int K = 1300;
//     int M = 1200;

//     size_t bytes_A = M * K * sizeof(float);
//     size_t bytes_B = K * N * sizeof(float);
//     size_t bytes_C = M * N * sizeof(float);

//     float *h_A, *h_B, *h_C;
//     h_A = (float*)malloc(bytes_A);
//     h_B = (float*)malloc(bytes_B); 
//     h_C = (float*)malloc(bytes_C);

//     float *d_A, *d_B, *d_C;
//     hipMalloc(&d_A, bytes_A);
//     hipMalloc(&d_B, bytes_B);
//     hipMalloc(&d_C, bytes_C);

//     for (int i = 0; i < M * K; i++) {
//         h_A[i] = i;
//     }

//     for (int i = 0; i < K * N; i++){
//         h_B[i] = i;
//     }

//     hipMemcpy(d_A, h_A, bytes_A, hipMemcpyHostToDevice);
//     hipMemcpy(d_B, h_B, bytes_B , hipMemcpyHostToDevice);

//     dim3 thread_per_block(16,16);
//     dim3 blocks_per_grid((N+15)/16, (M+15)/16);

//     matrix_mul<<<blocks_per_grid , thread_per_block >>>(d_A, d_B, d_C, M, N, K);

//     hipMemcpy(h_C, d_C, bytes_C, hipMemcpyDeviceToHost);

//     for (int i = 0; i < 3; i++) {
//     for (int j = 0; j < 3; j++) {
//         printf("%.2f ", h_C[i * N + j]);
//     }
//     printf("\n");
// }
//     return 0;
// }
