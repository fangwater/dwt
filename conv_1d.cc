#include "conv_1d.hpp"
#include <cstdio>
#include <immintrin.h>

__attribute__((target("avx2"))) static inline float hsum8_ps(__m256 v) {
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

__attribute__((target("default"))) int Conv1D::conv_8(float *data, float *res) {
    __m256 data_vec = _mm256_loadu_ps(data);
    __m256 mul = _mm256_mul_ps(data_vec, kernel_.get_kernel_vec_256()[0]);
    float sum = hsum8_ps(mul);
    *res = sum;
    constexpr int r = (sizeof(__m256) / sizeof(float))/8;
    return r;
}

__attribute__((target("avx512f"))) static inline float hsum16_ps(__m512 v) {
    __m256 vlow = _mm512_castps512_ps256(v);    // 低 256 位
    __m256 vhigh = _mm512_extractf32x8_ps(v, 1);// 高 256 位
    __m256 sum256 = _mm256_add_ps(vlow, vhigh);

    __m128 vlow128 = _mm256_castps256_ps128(sum256);   // 低 128 位
    __m128 vhigh128 = _mm256_extractf128_ps(sum256, 1);// 高 128 位
    __m128 sum128 = _mm_add_ps(vlow128, vhigh128);

    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 1));

    return _mm_cvtss_f32(sum128);
}

// AVX512 卷积函数
__attribute__((target("avx512f"))) int Conv1D::conv_8(float *data, float *res) {
    __m512 data_vec = _mm512_loadu_ps(data);
    __m512 kernel_vec = kernel_.get_kernel_vec_512()[0];
    __m512 mul = _mm512_mul_ps(data_vec, kernel_vec);
    float sum1 = hsum16_ps(mul);
    res[0] = sum1;
    __m512 data_vec_shifted = _mm512_loadu_ps(data + 1);
    __m512 mul_shifted = _mm512_mul_ps(data_vec_shifted, kernel_vec);
    float sum2 = hsum16_ps(mul_shifted);
    res[1] = sum2;
    constexpr int r = (sizeof(__m512) / sizeof(float)) / 8;
    return r;
}

// __attribute__((target("avx"))) static inline float hsum4_ps(__m128 v) {
//     // 将 __m128 的 4 个单精度浮点数求和
//     __m128 shuf = _mm_movehdup_ps(v); // 复制高半部分到低半部分
//     __m128 sums = _mm_add_ps(v, shuf);// 相加
//     shuf = _mm_movehl_ps(shuf, sums); // 将高半部分移动到低半部分
//     sums = _mm_add_ss(sums, shuf);    // 相加并得到最终结果
//     return _mm_cvtss_f32(sums);       // 提取结果
// }

// __attribute__((target("avx"))) void conv_12_simd256(float *data, float *kernel, float *res) {
//     // 加载前 8 个元素到 SIMD 寄存器
//     __m256 data_vec1 = _mm256_loadu_ps(data);
//     __m256 kernel_vec1 = _mm256_loadu_ps(kernel);

//     // 元素乘法
//     __m256 mul1 = _mm256_mul_ps(data_vec1, kernel_vec1);

//     // 加载后 4 个元素到 SIMD 寄存器
//     __m128 data_vec2 = _mm_loadu_ps(data + 8);
//     __m128 kernel_vec2 = _mm_loadu_ps(kernel + 8);

//     // 元素乘法
//     __m128 mul2 = _mm_mul_ps(data_vec2, kernel_vec2);

//     // 分别求和
//     float sum1 = hsum8_ps(mul1);
//     float sum2 = hsum4_ps(mul2);

//     // 合并结果并存储
//     *res = sum1 + sum2;
// }

// // 卷积核大小为16的卷积操作，步长为2
// void conv_16_simd256(float *data, size_t length, float *res) {
//     const size_t kernel_size = 16;
//     size_t res_index = 0;

//     for (size_t i = 0; i + kernel_size <= length; i += 2) {
//         float sum = 0.0f;
//         for (size_t j = 0; j < kernel_size; ++j) {
//             sum += data[i + j];
//         }
//         res[res_index++] = sum;
//     }
// }

// // 卷积核大小为30的卷积操作，步长为2
// void conv_30_simd256(float *data, size_t length, float *res) {
//     const size_t kernel_size = 30;
//     size_t res_index = 0;

//     for (size_t i = 0; i + kernel_size <= length; i += 2) {
//         float sum = 0.0f;
//         for (size_t j = 0; j < kernel_size; ++j) {
//             sum += data[i + j];
//         }
//         res[res_index++] = sum;
//     }
// }
