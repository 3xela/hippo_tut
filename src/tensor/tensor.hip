#include "tensor.h"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

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