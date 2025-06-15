#ifndef BACKWARD_H
#define BACKWARD_H


void launch_matmul_backward_A(float* grad_C, float* B, float* grad_A, int M, int N, int K) ;
void launch_matmul_backward_B(float* A, float* grad_C, float* grad_B, int M, int N, int K) ;
void launch_relu_backwards(float* input, float* output, int size) ;

#endif