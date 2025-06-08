#ifndef TRANSFORMER_H
#define TRANSFORMER_H

struct TransformerBlock {
    int d_model;
    int num_heads;
    int d_ff;
    
    float* W_q;
    float* W_k;
    float* W_v;
    float* W_o;
    float* W_ff1;
    float* W_ff2;
    
    float* ln1_gamma;
    float* ln1_beta;
    float* ln2_gamma;
    float* ln2_beta;
};

void transformer_block_forward(TransformerBlock* block, float* input, float* output, 
                              float* workspace, int seq_len, int num_heads);
void allocate_transformer_block(TransformerBlock* block, TransformerBlock* h_block);
void free_transformer_block(TransformerBlock* block);

#endif