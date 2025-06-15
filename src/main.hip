#include <hip/hip_runtime.h>
#include "tensor.h"
#include "kernels.h"
#include "backward.h"
#include <cstdlib>
#include <iostream>
#include <iomanip>

int main(){
    std::cout << "Testing ReLU Backward Implementation\n";
    std::cout << "====================================\n\n";

    size_t tensor_size = 10;  // Smaller size for easier verification

    // Create test tensor with known values
    Tensor* input_tensor = new Tensor(Shape({tensor_size}));
    Tensor* grad_output = new Tensor(Shape({tensor_size}));
    
    input_tensor->to_device();
    grad_output->to_device();

    // Set up test data: mix of positive, negative, and zero values
    float test_inputs[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f, -1.5f, 4.0f};
    float test_grad_outputs[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    
    // Copy test data to tensors
    for (size_t i = 0; i < tensor_size; i++) {
        input_tensor->data()[i] = test_inputs[i];
        grad_output->data()[i] = test_grad_outputs[i];
    }

    std::cout << "Input values:\n";
    for (size_t i = 0; i < tensor_size; i++) {
        std::cout << std::setw(6) << std::fixed << std::setprecision(1) 
                  << input_tensor->data()[i] << " ";
    }
    std::cout << "\n\n";

    // Test ReLU backward: gradient should be 1 for positive inputs, 0 for negative/zero
    launch_relu_backwards(input_tensor->data(), grad_output->data(), tensor_size);
    
    // Synchronize to ensure kernel completion
    hipDeviceSynchronize();
    
    grad_output->to_host();

    std::cout << "ReLU Backward Results:\n";
    std::cout << "Expected: [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]\n";
    std::cout << "Actual:   [";
    for (size_t i = 0; i < tensor_size; i++) {
        std::cout << (int)grad_output->data()[i];
        if (i < tensor_size - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";

    // Verify correctness
    bool all_correct = true;
    float expected[] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    
    for (size_t i = 0; i < tensor_size; i++) {
        if (grad_output->data()[i] != expected[i]) {
            all_correct = false;
            std::cout << "MISMATCH at index " << i << ": expected " 
                      << expected[i] << ", got " << grad_output->data()[i] << "\n";
        }
    }

    if (all_correct) {
        std::cout << "✅ ReLU Backward test PASSED!\n";
    } else {
        std::cout << "❌ ReLU Backward test FAILED!\n";
    }

    // Cleanup
    delete input_tensor;
    delete grad_output;
    
    return 0;
}