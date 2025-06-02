#include <hip/hip_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void tiled_matrix_mul(float* A, float* B, float* C, int M , int N, int K){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;


    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    for ( int t = 0 ; t< (K + TILE_SIZE -1)/TILE_SIZE; t++){
	int aRow = row;
	int aCol = t * TILE_SIZE + tx;
	
	int bRow = t * TILE_SIZE + ty;
   	int bCol = col;
	
	if (aRow < M && aCol < K)
		As[ty][tx] = A[aRow * K + aCol];
	else
		As[ty][tx] = 0.0f;
 	if (bRow < K && bCol <N)
		Bs[ty][tx] = B[bRow * N + bCol];
	else
		Bs[ty][tx] = 0.0f;
	__syncthreads();

    for (int k = 0; k < TILE_SIZE;k++){
		sum += As[ty][k] * Bs[k][tx];
	}
	__syncthreads();
    }

    if (row < M && col < N){
        C[row * N + col] = sum;
    }
}

int main(){
    int N = 1000;
    int K = 1300;
    int M = 1200;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(bytes_A);
    h_B = (float*)malloc(bytes_B); 
    h_C = (float*)malloc(bytes_C);

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes_A);
    hipMalloc(&d_B, bytes_B);
    hipMalloc(&d_C, bytes_C);

    for (int i = 0; i < M * K; i++) {
        h_A[i] = i;
    }

    for (int i = 0; i < K * N; i++){
        h_B[i] = i;
    }

    hipMemcpy(d_A, h_A, bytes_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, bytes_B , hipMemcpyHostToDevice);

    dim3 thread_per_block(TILE_SIZE, TILE_SIZE);
    dim3 blocks_per_grid((N+TILE_SIZE - 1)/ TILE_SIZE, (M+TILE_SIZE -1)/TILE_SIZE);

    tiled_matrix_mul<<<blocks_per_grid , thread_per_block >>>(d_A, d_B, d_C, M, N, K);

    hipMemcpy(h_C, d_C, bytes_C, hipMemcpyDeviceToHost);

    for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        printf("%.2f ", h_C[i * N + j]);
    }
    printf("\n");
}
    return 0;
}
