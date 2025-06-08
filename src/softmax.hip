#include <hip/hip_runtime.h>
#include <iostream>
#include <float.h>

__global__ void sum_vec(float* a, float* output){
	__shared__ float sdata[256];
	
	int tid = threadIdx.x;
	int	idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	sdata[tid] = a[idx];
	__syncthreads();

	for ( int stride = blockDim.x/2; stride >0; stride >>=1){

		if (idx < stride){
			sdata[idx] = fmaxf(sdata[idx], sdata[idx + stride]);
		}
		__syncthreads();
	}	
	
	if (tid == 0){
		output[blockIdx.x] = sdata[0];
	}
}	

__global__ void softmax_rows(float* input, float* output, int rows, int cols){
	__shared__ float sdata[256];
	
	int row = blockIdx.x;
	int col = threadIdx.x;
	int idx = row *cols + col;
	int tid = threadIdx.x;
	
	if (col < cols){
		sdata[tid] = input[idx];
	}
	else{
		sdata[tid] = -FLT_MAX;
	}
	__syncthreads();
	
	__shared__ float max_val;
	for ( int stride = blockDim.x/2; stride >0; stride >>=1){

		if (tid < stride){
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
		}
		__syncthreads();
	}	
	
	if (tid == 0){
		max_val = sdata[0];
	}
	__syncthreads();

	if ( col < cols) {
		sdata[tid] = expf(input[idx] - max_val);
	}
	else {
		sdata[tid] = 0.0f;
	}

	__syncthreads();

	for (int stride = blockDim.x/2; stride > 0; stride >>=1){
		if (tid < stride) {
			sdata[tid] += sdata[tid + stride];
		}
		__syncthreads();
	}
	__shared__ float sum_val;
	if (tid ==0){
		sum_val = sdata[0];
	}
	__syncthreads();

	if (col < cols){
		output[idx] = expf(input[idx] - max_val) / sum_val;
	}

}

__global__ void layer_norm(float* x, float* gamma, float* beta, float* y, int rows, int cols){
    __shared__ float sdata[256];
    __shared__ float mean;
    __shared__ float variance;
    
    int row = blockIdx.x;
    int col = threadIdx.x;
    int idx = row * cols + col;
    int tid = threadIdx.x;
    
    if (col < cols){
        sdata[tid] = x[idx];
    }
    else{
        sdata[tid] = 0.0f;  // changed from -FLT_MAX for mean calculation
    }
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if (tid < stride){
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0){
        mean = sdata[0] / cols;
    }
    __syncthreads();
    
    if (col < cols){
        sdata[tid] = powf(x[idx] - mean, 2);
    }
    else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0){
        variance = sdata[0] / cols;
    }
    __syncthreads();
    
    if (col < cols){
        y[idx] = gamma[col] * (x[idx] - mean) / sqrtf(variance + 1e-5f) + beta[col];
    }
}
void launch_softmax_rows(float* input, float* output, int rows, int cols){
	dim3 threads(256);
    dim3 blocks(rows);
	softmax_rows<<<blocks,threads>>>(input,output,rows,cols);
}

void launch_layer_norm(float* x, float* gamma, float* beta, float* y, int rows, int cols){
    dim3 threads(256);
    dim3 blocks(rows);
    layer_norm<<<blocks, threads>>>(x, gamma, beta, y, rows, cols);
}


// int main(){

// 	return 0;
// }
