#include "conv_kernel.hpp"
#include <cstdio>
void Conv1D(int kernel_size) {
}


#include <immintrin.h>


__attribute__((target("avx"))) static inline float hsum8_ps(__m256 v) {
    // 将 __m256 的 8 个单精度浮点数求和
    __m128 vlow = _mm256_castps256_ps128(v);   // 低 128 位
    __m128 vhigh = _mm256_extractf128_ps(v, 1);// 高 128 位
    vlow = _mm_add_ps(vlow, vhigh);            // 合并高低部分

    __m128 shuf = _mm_movehdup_ps(vlow);// 复制高半部分
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);// 移动高半部分到低半部分
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);// 提取结果
}


__attribute__((target("avx"))) void conv_8_simd256(float *data, float *kernel, float *res) {
    // 加载 data 和 kernel 到 SIMD 寄存器
    __m256 data_vec = _mm256_loadu_ps(data);
    __m256 kernel_vec = _mm256_loadu_ps(kernel);

    // 元素乘法
    __m256 mul = _mm256_mul_ps(data_vec, kernel_vec);

    // 求和
    float sum = hsum8_ps(mul);
    // 存储结果
    *res = sum;
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
