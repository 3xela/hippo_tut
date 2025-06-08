#include <hip/hip_runtime.h>
#include <iostream>
#include "kernels.h"

void launch_attention(float* Q, float* K, float* V, float* output, float* workspace, int seq_len, int d_model) {
    float scale = 1.0f / sqrtf((float)d_model);
    
    float* k_transpose = workspace;
    float* qk = workspace + seq_len * d_model;
    float* qk_scaled = qk + seq_len * seq_len;
    float* attn_weights = qk_scaled + seq_len * seq_len;
    
    launch_transpose(K, k_transpose, seq_len, d_model);
    
    launch_tiled_matmul(Q, k_transpose, qk, seq_len, seq_len, d_model);
    
    launch_scale(qk, qk_scaled, seq_len, seq_len, scale);
    
    launch_softmax_rows(qk_scaled, attn_weights, seq_len, seq_len);
    
    launch_tiled_matmul(attn_weights, V, output, seq_len, d_model, seq_len);
}

void launch_multihead_attention(float* Q, float* K, float* V, float* output, 
    float* workspace, int seq_len, int d_model, int num_heads) {
    int d_k = d_model / num_heads;

for (int h = 0; h < num_heads; h++) {
    float* Q_h = Q + h * d_k;
    float* K_h = K + h * d_k;
    float* V_h = V + h * d_k;
    float* out_h = output + h * d_k;

        launch_attention(Q_h, K_h, V_h, out_h, workspace, seq_len, d_k);
    }
}