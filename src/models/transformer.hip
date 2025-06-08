#include <hip/hip_runtime.h>
#include "kernels.h"
#include "tensor.h"
#include <iostream>

struct TransformerBlock {
    int d_model;
    int num_heads;
    int d_ff;  // feedforward hidden size, typically 4 * d_model
    
    // weights
    float* W_q;      // [d_model, d_model]
    float* W_k;      // [d_model, d_model]
    float* W_v;      // [d_model, d_model]
    float* W_o;      // [d_model, d_model]
    float* W_ff1;    // [d_model, d_ff]
    float* W_ff2;    // [d_ff, d_model]
    
    // layer norm params
    float* ln1_gamma;  // [d_model]
    float* ln1_beta;   // [d_model]
    float* ln2_gamma;  // [d_model]
    float* ln2_beta;   // [d_model]
};

void transformer_block_forward(TransformerBlock* block, float* input, float* output, 
                              float* workspace, int seq_len, int num_heads) {
    float* ln1_out = workspace;
    float* q = ln1_out + seq_len * block->d_model;
    float* k = q + seq_len * block->d_model;
    float* v = k + seq_len * block->d_model;
    float* attn_out = v + seq_len * block->d_model;
    float* attn_proj = attn_out + seq_len * block->d_model;
    float* residual1 = attn_proj + seq_len * block->d_model;
    float* ln2_out = residual1 + seq_len * block->d_model;
    float* ff1_out = ln2_out + seq_len * block->d_model;
    float* ff1_act = ff1_out + seq_len * block->d_ff;
    float* ff2_out = ff1_act + seq_len * block->d_ff;
    
    // 1. layer norm
    launch_layer_norm(input, block->ln1_gamma, block->ln1_beta, ln1_out, seq_len, block->d_model);
    
    // 2. generate Q, K, V
    launch_tiled_matmul(ln1_out, block->W_q, q, seq_len, block->d_model, block->d_model);
    launch_tiled_matmul(ln1_out, block->W_k, k, seq_len, block->d_model, block->d_model);
    launch_tiled_matmul(ln1_out, block->W_v, v, seq_len, block->d_model, block->d_model);
    
    // 3. multi-head attention (for now use single head)
    float* attn_workspace = ff2_out + seq_len * block->d_model;  // reuse space
    launch_multihead_attention(q, k, v, attn_out, attn_workspace, seq_len, block->d_model, num_heads);
    
    // 4. output projection
    launch_tiled_matmul(attn_out, block->W_o, attn_proj, seq_len, block->d_model, block->d_model);
    
    // 5. residual add
    launch_add_residual(input, attn_proj, residual1, seq_len * block->d_model);
    
    // 6. layer norm 2
    launch_layer_norm(residual1, block->ln2_gamma, block->ln2_beta, ln2_out, seq_len, block->d_model);
    
    // 7. feedforward
    launch_tiled_matmul(ln2_out, block->W_ff1, ff1_out, seq_len, block->d_ff, block->d_model);
    launch_relu(ff1_out, ff1_act, seq_len * block->d_ff);
    launch_tiled_matmul(ff1_act, block->W_ff2, ff2_out, seq_len, block->d_model, block->d_ff);
    
    // 8. final residual
    launch_add_residual(residual1, ff2_out, output, seq_len * block->d_model);
}
float* to_device(float* host_data, size_t size) {
    float* device_data;
    hipMalloc(&device_data, size * sizeof(float));
    hipMemcpy(device_data, host_data, size * sizeof(float), hipMemcpyHostToDevice);
    return device_data;
}

void allocate_transformer_block(TransformerBlock* block, TransformerBlock* h_block) {
    block->d_model = h_block->d_model;
    block->num_heads = h_block->num_heads;
    block->d_ff = h_block->d_ff;
    
    size_t weight_size = block->d_model * block->d_model;
    size_t ff1_size = block->d_model * block->d_ff;
    size_t ff2_size = block->d_ff * block->d_model;
    
    block->W_q = to_device(h_block->W_q, weight_size);
    block->W_k = to_device(h_block->W_k, weight_size);
    block->W_v = to_device(h_block->W_v, weight_size);
    block->W_o = to_device(h_block->W_o, weight_size);
    block->W_ff1 = to_device(h_block->W_ff1, ff1_size);
    block->W_ff2 = to_device(h_block->W_ff2, ff2_size);
    
    block->ln1_gamma = to_device(h_block->ln1_gamma, block->d_model);
    block->ln1_beta = to_device(h_block->ln1_beta, block->d_model);
    block->ln2_gamma = to_device(h_block->ln2_gamma, block->d_model);
    block->ln2_beta = to_device(h_block->ln2_beta, block->d_model);
}

void free_transformer_block(TransformerBlock* block) {
    hipFree(block->W_q);
    hipFree(block->W_k);
    hipFree(block->W_v);
    hipFree(block->W_o);
    hipFree(block->W_ff1);
    hipFree(block->W_ff2);
    hipFree(block->ln1_gamma);
    hipFree(block->ln1_beta);
    hipFree(block->ln2_gamma);
    hipFree(block->ln2_beta);
}