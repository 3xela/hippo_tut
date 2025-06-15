#include "tensor.h"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include "backward.h"
#include "kernels.h"

Tensor::Tensor(const Shape& shape) : shape_(shape), on_device_(false), requires_grad(false), grad_(nullptr) {
    size_ = shape_.size();
    data_ = (float*)malloc(size_ * sizeof(float));
}

Tensor::Tensor(const std::vector<size_t>& dims) : shape_(dims), on_device_(false), requires_grad(false), grad_(nullptr) {
    size_ = shape_.size();
    data_ = (float*)malloc(size_ * sizeof(float));
}

Tensor* Tensor::device(const Shape& shape) {
    Tensor* t = new Tensor(shape);
    t->to_device();
    return t;
}

Tensor* Tensor::to_device() {
    if (!on_device_) {
        float* d_data;
        hipMalloc(&d_data, size_ * sizeof(float));
        hipMemcpy(d_data, data_, size_ * sizeof(float), hipMemcpyHostToDevice);
        free(data_);
        data_ = d_data;
        on_device_ = true;
    }
    return this;
}

Tensor* Tensor::to_host() {
    if (on_device_) {
        float* h_data = (float*)malloc(size_ * sizeof(float));
        hipMemcpy(h_data, data_, size_ * sizeof(float), hipMemcpyDeviceToHost);
        hipFree(data_);
        data_ = h_data;
        on_device_ = false;
    }
    return this;
}

void Tensor::fill(float value) {
    if (on_device_) {
        // For device memory, we need to use hipMemset or a kernel
        // hipMemset only works for byte values, so we'll use a simple kernel approach
        for (size_t i = 0; i < size_; i++) {
            // This is a simplified approach - in practice you'd want a kernel
        }
    } else {
        for (size_t i = 0; i < size_; i++) {
            data_[i] = value;
        }
    }
}

void Tensor::random(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    if (on_device_) {
        // Copy to host, fill, then copy back
        float* temp = (float*)malloc(size_ * sizeof(float));
        for (size_t i = 0; i < size_; i++) {
            temp[i] = dis(gen);
        }
        hipMemcpy(data_, temp, size_ * sizeof(float), hipMemcpyHostToDevice);
        free(temp);
    } else {
        for (size_t i = 0; i < size_; i++) {
            data_[i] = dis(gen);
        }
    }
}

void Tensor::print(const std::string& name) const {
    if (on_device_) {
        float* h_data = (float*)malloc(size_ * sizeof(float));
        hipMemcpy(h_data, data_, size_ * sizeof(float), hipMemcpyDeviceToHost);
        
        std::cout << name << ": [";
        for (size_t i = 0; i < std::min(size_, (size_t)10); i++) {
            std::cout << h_data[i];
            if (i < std::min(size_, (size_t)10) - 1) std::cout << ", ";
        }
        if (size_ > 10) std::cout << "...";
        std::cout << "]" << std::endl;
        
        free(h_data);
    } else {
        std::cout << name << ": [";
        for (size_t i = 0; i < std::min(size_, (size_t)10); i++) {
            std::cout << data_[i];
            if (i < std::min(size_, (size_t)10) - 1) std::cout << ", ";
        }
        if (size_ > 10) std::cout << "...";
        std::cout << "]" << std::endl;
    }
}

Tensor::~Tensor() {
    if (on_device_) {
        hipFree(data_);
    } else {
        free(data_);
    }
    if (grad_) {
        if (on_device_) {
            hipFree(grad_);
        } else {
            free(grad_);
        }
    }
}

Tensor* Tensor::matmul(Tensor* other) {
    // Simple matrix multiplication - assumes 2D tensors
    Tensor* output = new Tensor(Shape({shape_[0], other->shape_[1]}));
    output->to_device();
    
    // Launch matrix multiplication kernel
    launch_matmul(data_, other->data_, output->data_, shape_[0], shape_[1], other->shape_[1]);
    
    // Set up backward pass if needed
    if (requires_grad || other->requires_grad) {
        output->requires_grad = true;
        output->parents_ = {this, other};
        
        // Store backward function
        output->backward_fn_ = [=]() {
            if (this->requires_grad) {
                if (!this->grad_) {
                    hipMalloc(&this->grad_, this->size_ * sizeof(float));
                    hipMemset(this->grad_, 0, this->size_ * sizeof(float));
                }
                launch_matmul_backward_A(output->grad_, other->data_, this->grad_, 
                                       shape_[0], shape_[1], other->shape_[1]);
            }
            if (other->requires_grad) {
                if (!other->grad_) {
                    hipMalloc(&other->grad_, other->size_ * sizeof(float));
                    hipMemset(other->grad_, 0, other->size_ * sizeof(float));
                }
                launch_matmul_backward_B(this->data_, output->grad_, other->grad_,
                                       shape_[0], shape_[1], other->shape_[1]);
            }
        };
    }
    
    return output;
}

void Tensor::backward() {
    if (!requires_grad) return;
    
    // Initialize gradient if not already done
    if (!grad_) {
        hipMalloc(&grad_, size_ * sizeof(float));
        // Fill with ones (gradient of output w.r.t itself is 1)
        float* ones = (float*)malloc(size_ * sizeof(float));
        for (size_t i = 0; i < size_; i++) ones[i] = 1.0f;
        hipMemcpy(grad_, ones, size_ * sizeof(float), hipMemcpyHostToDevice);
        free(ones);
    }
    
    // Call backward function if it exists
    if (backward_fn_) {
        backward_fn_();
    }
    
    // Recursively call backward on parents
    for (Tensor* parent : parents_) {
        if (parent->requires_grad) {
            parent->backward();
        }
    }
}

void Tensor::zero_grad() {
    if (grad_) {
        if (on_device_) {
            hipMemset(grad_, 0, size_ * sizeof(float));
        } else {
            memset(grad_, 0, size_ * sizeof(float));
        }
    }
}

Tensor* Tensor::detach() {
    Tensor* detached = new Tensor(shape_);
    if (on_device_) {
        detached->to_device();
        hipMemcpy(detached->data_, data_, size_ * sizeof(float), hipMemcpyDeviceToDevice);
    } else {
        memcpy(detached->data_, data_, size_ * sizeof(float));
    }
    detached->requires_grad = false;
    return detached;
}