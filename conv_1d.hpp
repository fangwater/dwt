#ifndef CONV_1D_HPP
#define CONV_1D_HPP
#include <functional>
#include <immintrin.h>
#include <stdexcept>
#include "kernel.hpp"
#include <fmt/format.h>
inline void print_vector(const std::vector<float> &vec) {
    fmt::print("[");
    for (size_t i = 0; i < vec.size(); ++i) {
        fmt::print("{:.3f}", vec[i]);
        if (i < vec.size() - 1) {
            fmt::print(", ");
        }
    }
    fmt::print("]\n");
}
class Conv1D {
public:
    using ConvFunction = std::function<int(float *, float *)>;
    Conv1D(const float* filter, int window_size, int stride) : stride_(stride),kernel_(filter, window_size){
        switch (window_size) {
            case 8:
                conv_func_ = [this](float *data, float *res) { return this->conv_8(data, res); };
                break;
            case 12:
                conv_func_ = [this](float *data, float *res) { return this->conv_12(data, res); };
                break;
            case 16:
                conv_func_ = [this](float *data, float *res) { return this->conv_16(data, res); };
                break;
            case 30:
                conv_func_ = [this](float *data, float *res) { return this->conv_30(data, res); };
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
    int conv_12(float *data, float *res) __attribute__((target("default")));
    int conv_16(float *data, float *res) __attribute__((target("default")));
    int conv_30(float *data, float *res) __attribute__((target("default")));
    // 卷积函数 avx512版本
    int conv_8_avx512(float *data, float *res) __attribute__((target("avx512f")));
    int conv_12(float *data, float *res) __attribute__((target("avx512f")));
    int conv_16(float *data, float *res) __attribute__((target("avx512f")));
    int conv_30(float *data, float *res) __attribute__((target("avx512f")));
    Kernel kernel_;
    // 使用 std::function 代替函数指针
    ConvFunction conv_func_;
    int stride_;
};
#endif