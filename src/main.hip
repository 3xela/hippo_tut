#include <hip/hip_runtime.h>
#include "transformers.h"
#include "tensor.h"

int main() {
    // model config
    size_t seq_len = 128;
    size_t d_model = 512;
    size_t num_heads = 8;
    size_t d_ff = 2048;
    
    // create transformer block on host
    TransformerBlock h_block;
    h_block.d_model = d_model;
    h_block.num_heads = num_heads;
    h_block.d_ff = d_ff;
    
    // initialize weights on host (random for now)
    h_block.W_q = (float*)malloc(d_model * d_model * sizeof(float));
    h_block.W_k = (float*)malloc(d_model * d_model * sizeof(float));
    h_block.W_v = (float*)malloc(d_model * d_model * sizeof(float));
    h_block.W_o = (float*)malloc(d_model * d_model * sizeof(float));
    h_block.W_ff1 = (float*)malloc(d_model * d_ff * sizeof(float));
    h_block.W_ff2 = (float*)malloc(d_ff * d_model * sizeof(float));
    h_block.ln1_gamma = (float*)malloc(d_model * sizeof(float));
    h_block.ln1_beta = (float*)malloc(d_model * sizeof(float));
    h_block.ln2_gamma = (float*)malloc(d_model * sizeof(float));
    h_block.ln2_beta = (float*)malloc(d_model * sizeof(float));
    
    // simple initialization
    for (size_t i = 0; i < d_model * d_model; i++) {
        h_block.W_q[i] = 0.02f * (rand() / (float)RAND_MAX - 0.5f);
        h_block.W_k[i] = 0.02f * (rand() / (float)RAND_MAX - 0.5f);
        h_block.W_v[i] = 0.02f * (rand() / (float)RAND_MAX - 0.5f);
        h_block.W_o[i] = 0.02f * (rand() / (float)RAND_MAX - 0.5f);
    }
    
    for (size_t i = 0; i < d_model; i++) {
        h_block.ln1_gamma[i] = 1.0f;
        h_block.ln1_beta[i] = 0.0f;
        h_block.ln2_gamma[i] = 1.0f;
        h_block.ln2_beta[i] = 0.0f;
    }
    
    // move to device
    TransformerBlock d_block;
    allocate_transformer_block(&d_block, &h_block);
    
    // create input tensor
    Tensor* input = new Tensor(Shape({seq_len * d_model}));
    for (size_t i = 0; i < seq_len * d_model; i++) {
        input->data()[i] = 0.1f * (rand() / (float)RAND_MAX);
    }
    input->to_device();
    
    // output tensor
    Tensor* output = Tensor::device(Shape({seq_len * d_model}));
    
    // workspace
    size_t workspace_size = seq_len * d_model * 10 + seq_len * seq_len * 4 + seq_len * d_ff * 2;
    Tensor* workspace = Tensor::device(Shape({workspace_size}));
    
    // run forward pass
    transformer_block_forward(&d_block, input->data(), output->data(), workspace->data(), seq_len, num_heads);
    
    // get results
    output->to_host();
    
    // print some outputs
    printf("Output (first 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", output->data()[i]);
    }
    printf("\n");
    
    // cleanup
    delete input;
    delete output;
    delete workspace;
    free_transformer_block(&d_block);
    
    // free host memory
    free(h_block.W_q);
    free(h_block.W_k);
    // ... etc
    
    return 0;
}