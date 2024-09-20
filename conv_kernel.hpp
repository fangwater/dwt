#ifndef CONV_KERNEL
#define CONV_KERNEL
#include <cstddef>
#include <functional>
#include <immintrin.h>
#include "kernel.hpp"

class Conv1D {
public:
    using ConvFunction = std::function<void(float *, float *)>;
    Conv1D(float *kernel,int window_size);
    void convolve(float *data, float *res);
private:
    void initialize();
    // 卷积函数
    void conv_8_simd256(float *data, float *res);
    void conv_12_simd256(float *data, float *res);
    void conv_16_simd256(float *data, float *res);
    void conv_30_simd256(float *data, float *res);
    void conv_generic(float *data, float *res);
    Kernel kernel_;
    // 使用 std::function 代替函数指针
    ConvFunction conv_func_;
};

Conv1D::Conv1D(float *kernel, int window_size):kernel_(kernel,window_size){
    
}

//8,12,16,30
void conv_8_simd256(float *data, float *kernel, float *res);

// 卷积核大小为12的卷积操作，步长为2
void conv_12_simd256(float *data, size_t length, float *res);

// 卷积核大小为16的卷积操作，步长为2
void conv_16_simd256(float *data, size_t length, float *res);

// 卷积核大小为30的卷积操作，步长为2
void conv_30_simd256(float *data, size_t length, float *res);


#endif