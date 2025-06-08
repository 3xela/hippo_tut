#include <hip/hip_runtime.h>
#include <iostream>

__global__ void add_vec(int* a, int* b, int* c , int n ){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        c[idx] = a[idx] + b[idx];
    }
}
// int main(){
//     int n = 1000;
//     size_t bytes = n * sizeof(bytes);

//     int *h_a, *h_b, *h_c;
//     h_a = (int*)malloc(bytes);
//     h_b = (int*)malloc(bytes);
//     h_c = (int*)malloc(bytes);

//     int *d_a, *d_b, *d_c;
//     hipMalloc(&d_a, bytes);
//     hipMalloc(&d_b, bytes);
//     hipMalloc(&d_c, bytes);


//     for (int i=0; i<n; i++){
//         h_a[i] = i;
//         h_b[i] = i;        
//     }
//     hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
//     hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

//     int threads_per_block = 256;

//     int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
//     add_vec<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);      

//     hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

//     // print out some values to make sure the addition worked
//     for (int i = 0; i<10; i++){
//         printf("value at index %d: %d\n " , i, h_c[i]);
//     }
//     hipFree(d_a);
//     hipFree(d_b);
//     hipFree(d_c);

//     printf("freed hip memory \n");

//     free(h_a);
//     free(h_b);
//     free(h_c);
    
//     printf("freed cpu memory \n");


//     return 0;
// }

