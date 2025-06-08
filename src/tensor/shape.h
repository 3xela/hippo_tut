#ifndef SHAPE_H
#define SHAPE_H

#include <vector>
#include <cstddef>  
#include <stdexcept>
class Shape {
private:
    std::vector<size_t> dims_;
    
public:
    Shape(const std::vector<size_t>& dims) : dims_(dims) {}
    
    size_t size() const {
        size_t cumprod = 1;
        for(int i = 0 ; i < dims_.size(); i++){
            cumprod *= dims_[i];
        }
        return cumprod;
    }
    
    size_t ndim() const {
        return dims_.size();
    }
    
    size_t operator[](int index) const {
        return dims_[index];
    }

    std::vector<size_t> strides() const {
        std::vector<size_t> s(dims_.size());
        s[dims_.size() - 1] = 1;
        for (int i = dims_.size() - 2; i >= 0; i--) {
            s[i] = s[i + 1] * dims_[i + 1];
        }
        return s;
    }
    Shape flatten() const {
        return Shape({size()});
    }
    Shape squeeze() const{
        std::vector<size_t> new_dims;
        for (size_t dim : dims_) {
            if (dim != 1) {
                new_dims.push_back(dim);
            }
        }
        return Shape(new_dims);
    }
    Shape unsqueeze(int dim){
        std::vector<size_t> new_dims = dims_;
        new_dims.insert(new_dims.begin() + dim, 1);
        return Shape(new_dims);
    }

    Shape transpose(const std::vector<int>& axes = {}) const {
        std::vector<size_t> new_dims = dims_;

        if (axes.empty()){
            new_dims = dims_;
            if (dims_.size() >= 2) {
                std::swap(new_dims[dims_.size()-2], new_dims[dims_.size()-1]);
            }
        }else{
        for (int i = 0; i < axes.size(); i++ ){
            new_dims[i] = dims_[axes[i]];
        }
    }
    return Shape(new_dims);
    }

    Shape reshape(const std::vector<int>& new_shape) const {
        size_t new_total = 1;
        int infer_idx = -1;
        
        for (int i = 0; i < new_shape.size(); i++) {
            if (new_shape[i] == -1) {
                infer_idx = i;
            } else {
                new_total *= new_shape[i];
            }
        }
        
        std::vector<size_t> final_shape;
        for (int i = 0; i < new_shape.size(); i++) {
            if (i == infer_idx) {
                final_shape.push_back(size() / new_total);
            } else {
                final_shape.push_back(new_shape[i]);
            }
        }
        
        // Validate
        if (Shape(final_shape).size() != size()) {
            throw std::invalid_argument("Incompatible reshape");
        }
        
        return Shape(final_shape);
    }


};

#endif