#include "conv_kernel.hpp"
void Conv1D(int kernel_size) {
}


//8,12,16,30
void conv_8_simd256(float *data, size_t length, float *res) {
    const size_t kernel_size = 8;
    size_t res_index = 0;

    // 进行卷积操作，步长为2
    for (size_t i = 0; i + kernel_size <= length; i += 2) {
        float sum = 0.0f;
        // 累加卷积核与数据的乘积
        for (size_t j = 0; j < kernel_size; ++j) {
            sum += data[i + j];
        }
        // 将结果保存到输出数组
        res[res_index++] = sum;
    }
}

// 卷积核大小为12的卷积操作，步长为2
void conv_12_simd256(float *data, size_t length, float *res) {
    const size_t kernel_size = 12;
    size_t res_index = 0;

    for (size_t i = 0; i + kernel_size <= length; i += 2) {
        float sum = 0.0f;
        for (size_t j = 0; j < kernel_size; ++j) {
            sum += data[i + j];
        }
        res[res_index++] = sum;
    }
}

// 卷积核大小为16的卷积操作，步长为2
void conv_16_simd256(float *data, size_t length, float *res) {
    const size_t kernel_size = 16;
    size_t res_index = 0;

    for (size_t i = 0; i + kernel_size <= length; i += 2) {
        float sum = 0.0f;
        for (size_t j = 0; j < kernel_size; ++j) {
            sum += data[i + j];
        }
        res[res_index++] = sum;
    }
}

// 卷积核大小为30的卷积操作，步长为2
void conv_30_simd256(float *data, size_t length, float *res) {
    const size_t kernel_size = 30;
    size_t res_index = 0;

    for (size_t i = 0; i + kernel_size <= length; i += 2) {
        float sum = 0.0f;
        for (size_t j = 0; j < kernel_size; ++j) {
            sum += data[i + j];
        }
        res[res_index++] = sum;
    }
}
