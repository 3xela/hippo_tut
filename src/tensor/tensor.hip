#include "tensor.h"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

Tensor::Tensor(const Shape& shape) : shape_(shape), on_device_(false), requires_grad(false), grad_(nullptr) {
    size_ = shape_.size();
    data_ = (float*)malloc(size_ * sizeof(float));
}

Tensor::Tensor(const std::vector<size_t>& dims) : shape_(dims), on_device_(false), requires_grad(false), grad_(nullptr) {
    size_ = shape_.size();
    data_ = (float*)malloc(size_ * sizeof(float));
}

Tensor::Tensor(const Shape& shape) : shape_(shape), on_device_(false) {
    size_ = shape_.size();
    data_ = (float*)malloc(size_ * sizeof(float));
}

Tensor::Tensor(const std::vector<size_t>& dims) : shape_(dims), on_device_(false) {
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

Tensor* Tensor::reshape(const Shape& new_shape) {
    if (new_shape.size() != size_) {
        throw std::invalid_argument("Reshape size mismatch");
    }
    shape_ = new_shape;
    return this;
}

Tensor* Tensor::view(const std::vector<size_t>& dims) {
    return reshape(Shape(dims));
}

Tensor* Tensor::slice(size_t start, size_t end, int dim) {
    // TODO: implement slicing
    return this;
}

void Tensor::fill(float value) {
    if (on_device_) {
        hipMemset(data_, value, size_ * sizeof(float));
    } else {
        for (size_t i = 0; i < size_; i++) {
            data_[i] = value;
        }
    }
}

void Tensor::random(float min, float max) {
    if (on_device_) {
        float* temp = (float*)malloc(size_ * sizeof(float));
        for (size_t i = 0; i < size_; i++) {
            temp[i] = min + (max - min) * (rand() / (float)RAND_MAX);
        }
        hipMemcpy(data_, temp, size_ * sizeof(float), hipMemcpyHostToDevice);
        free(temp);
    } else {
        for (size_t i = 0; i < size_; i++) {
            data_[i] = min + (max - min) * (rand() / (float)RAND_MAX);
        }
    }
}

void Tensor::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << ":\n";
    }
    
    float* h_data = data_;
    if (on_device_) {
        h_data = (float*)malloc(size_ * sizeof(float));
        hipMemcpy(h_data, data_, size_ * sizeof(float), hipMemcpyDeviceToHost);
    }
    
    // Print first few elements
    size_t print_size = std::min(size_, size_t(10));
    for (size_t i = 0; i < print_size; i++) {
        std::cout << h_data[i] << " ";
    }
    if (size_ > 10) std::cout << "...";
    std::cout << "\n";
    
    if (on_device_) {
        free(h_data);
    }
}

Tensor::~Tensor() {
    if (on_device_) {
        hipFree(data_);
    } else {
        free(data_);
    }
}
Tensor* Tensor::matmul(Tensor* other) {
    Tensor* output = new Tensor(Shape({shape_[0], other->shape_[1]}));
    launch_matmul(data_, other->data_, output->data_, ...);
    
    if (requires_grad_ || other->requires_grad) {
        output->requires_grad = true;
        output->parents_ = {this, other};
        output->backward_fn_ = [=]() {
            // grad_input = grad_output @ other^T
            launch_matmul(output->grad_, other->data_, this->grad_, ...);
            // grad_other = input^T @ grad_output  
            launch_matmul(this->data_, output->grad_, other->grad_, ...);
        };
    }
    return output;
}
void Tensor::backward() {
    if (!grad_) {
        grad_ = allocate_like(data_);
        fill_ones(grad_);  // gradient of output w.r.t itself is 1
    }
    
    if (backward_fn_) {
        backward_fn_();  // compute gradients for parents
        for (auto parent : parents_) {
            parent->backward();  // recursive
        }
    }
}

Tensor* Tensor::matmul(Tensor* other) {
    // Shape check
    if (shape_[shape_.ndim()-1] != other->shape_[shape_.ndim()-2]) {
        throw std::invalid_argument("Matmul dimension mismatch");
    }
    
    // Create output tensor
    std::vector<size_t> out_dims = shape_.dims();
    out_dims[out_dims.size()-1] = other->shape_[shape_.ndim()-1];
    Tensor* output = new Tensor(Shape(out_dims));
    
    // Move to device if needed
    if (on_device_) output->to_device();
    
    // Forward pass
    launch_matmul(data_, other->data_, output->data_, 
                  shape_[0], other->shape_[1], shape_[1]);
    
    // Setup backward pass if needed
    if (requires_grad || other->requires_grad) {
        output->requires_grad = true;
        output->parents_ = {this, other};
        
        output->backward_fn_ = [this, other, output]() {
            if (this->requires_grad) {
                if (!this->grad_) {
                    this->grad_ = (float*)hipMalloc(this->size_ * sizeof(float));
                    hipMemset(this->grad_, 0, this->size_ * sizeof(float));
                }
                // grad_input = grad_output @ other^T
                launch_matmul_backward_input(output->grad_, other->data_, this->grad_,
                                           output->shape_[0], output->shape_[1], other->shape_[0]);
            }
            if (other->requires_grad) {
                if (!other->grad_) {
                    other->grad_ = (float*)hipMalloc(other->size_ * sizeof(float));
                    hipMemset(other->grad_, 0, other->size_ * sizeof(float));
                }
                // grad_other = input^T @ grad_output
                launch_matmul_backward_weight(this->data_, output->grad_, other->grad_,
                                            this->shape_[1], this->shape_[0], output->shape_[1]);
            }
        };
    }
    
    return output;
}