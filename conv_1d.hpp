#ifndef CONV_1D_HPP
#define CONV_1D_HPP
#include <functional>
#include <immintrin.h>
#include <stdexcept>
#include "kernel.hpp"

class Conv1D {
public:
    using ConvFunction = std::function<int(float *, float *)>;
    Conv1D(const float* filter, int window_size, int stride) : stride_(stride),kernel_(filter, window_size){
        switch (window_size) {
            case 8:
                conv_func_ = [this](float *data, float *res) { return this->conv_8(data, res); };
                break;
            default:
                throw std::invalid_argument("Unsupported window size");
        }
    }
    int conv_1d(float *data, float *res) {
        return conv_func_(data,res);
    }
public:
    // 卷积函数 avx2版本
    int conv_8(float *data, float *res) __attribute__((target("default")));
    // void conv_12(float *data, float *res) __attribute__((target("default")));
    // void conv_16(float *data, float *res) __attribute__((target("default")));
    // void conv_30(float *data, float *res) __attribute__((target("default")));
    // 卷积函数 avx512版本
    int conv_8_avx512(float *data, float *res) __attribute__((target("avx512f")));
    // void conv_12(float *data, float *res) __attribute__((target("avx512f")));
    // void conv_16(float *data, float *res) __attribute__((target("avx512f")));
    // void conv_30(float *data, float *res) __attribute__((target("avx512f")));
    Kernel kernel_;
    // 使用 std::function 代替函数指针
    ConvFunction conv_func_;
    int stride_;
};
#endif