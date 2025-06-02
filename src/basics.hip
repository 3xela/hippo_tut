#include <hip/hip_runtime.h>
#include <iostream>

__global__ void add(int* a, int* b, int* c, int n) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < n) {
    printf("Block Id: %d \n", blockIdx.x);
    printf("Thread Id: %d \n", threadIdx.x);
    }
}

int main() {
    int a = 10;
    int* ptr = &a;
    int size = *ptr;

    add<<<2,32>>>(&size, &size, &size, 100);
    hipDeviceSynchronize();
    return 0;
}