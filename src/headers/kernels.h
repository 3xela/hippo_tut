#ifndef KERNELS_H
#define KERNELS_H

void launch_matmul(float* A, float* B, float* C, int M, int N, int K);
void launch_tiled_matmul(float* A, float* B, float* C, int M, int N, int K);
void launch_softmax_rows(float* input, float* output, int rows, int cols);
void launch_transpose(float* input, float* output, int rows, int cols);
void launch_layer_norm(float* x, float* gamma, float* beta, float* y, int rows, int cols);
void launch_transpose(float* A, float* B, int M, int N);
void launch_scale(float* A, float* B, int M, int N, float scale);
void launch_attention(float* Q, float* K, float* V, float* output, float* workspace, int seq_len, int d_model);
void launch_relu(float* x, float* output, int size);
void launch_multihead_attention(float* Q, float* K, float* V, float* output, 
    float* workspace, int seq_len, int d_model, int num_heads);
void launch_add_residual(float* x, float* residual, float* output, int size);
#endif