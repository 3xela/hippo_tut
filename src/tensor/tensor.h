#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include "shape.h"

class Tensor {
private:
    float* data_;
    Shape shape_;
    size_t size_;
    bool on_device_;
    
public:
    // Constructors
    Tensor(const Shape& shape);
    Tensor(const std::vector<size_t>& dims);
    static Tensor* device(const Shape& shape);
    
    // Memory management
    Tensor* to_device();
    Tensor* to_host();
    
    // Access
    float* data() { return data_; }
    const Shape& shape() const { return shape_; }
    size_t size() const { return size_; }
    bool is_device() const { return on_device_; }
    
    // Operations
    Tensor* reshape(const Shape& new_shape);
    Tensor* view(const std::vector<size_t>& dims);
    Tensor* slice(size_t start, size_t end, int dim = 0);
    
    // Utilities
    void fill(float value);
    void random(float min = 0.0f, float max = 1.0f);
    void print(const std::string& name = "") const;
    
    ~Tensor();
};

#endif