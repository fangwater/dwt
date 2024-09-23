#include "conv_1d.hpp"
#include <cstdio>
#include <immintrin.h>

__attribute__((target("avx2"))) inline void horizontal_sum_m256(__m256 *x, float *res) {
    const __m128 hiQuad = _mm256_extractf128_ps(*x, 1);
    const __m128 loQuad = _mm256_castps256_ps128(*x);
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    const __m128 loDual = sumQuad;
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    const __m128 lo = sumDual;
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    const __m128 sum = _mm_add_ss(lo, hi);
    *res = _mm_cvtss_f32(sum);
}

__attribute__((target("avx512f"))) inline void hadd_m512_halves(__m512 *input, float *lower_res, float *upper_res) {
    __m256 lower_half = _mm512_extractf32x8_ps(*input, 0);
    __m256 upper_half = _mm512_extractf32x8_ps(*input, 1);
    horizontal_sum_m256(&lower_half, lower_res);
    horizontal_sum_m256(&upper_half, upper_res);
}

__attribute__((target("avx512f"))) __m512 load_floats_to_m512(const float *lower_half, const float *upper_half) {
    // 使用 _mm256_loadu_ps 一次性从 lower_half 和 upper_half 各加载8个float
    __m256 lower_part = _mm256_loadu_ps(lower_half);// 加载前8个float到__m256
    __m256 upper_part = _mm256_loadu_ps(upper_half);// 加载后8个float到__m256

    __m512 result = _mm512_castps256_ps512(lower_part);// 将lower_part转换为__m512的前半部分
    result = _mm512_insertf32x8(result, upper_part, 1);// 将upper_part插入到__m512的后半部分

    return result;
}

/**
 * @brief avx256的情况下，对window = 8进行卷积
 */
__attribute__((target("default"))) int Conv1D::conv_8(float *data, float *res) {
    __m256 data_vec = _mm256_loadu_ps(data);
    __m256 mul = _mm256_mul_ps(data_vec, kernel_.get_kernel_vec_256()[0]);
    horizontal_sum_m256(&mul, res);
    constexpr int r = (sizeof(__m256) / sizeof(float))/8;
    return r;
}

/**
 * @brief avx512的情况下，对window = 8进行卷积 需要读取16个float 产生两个res
 * 此时，两个window的数据不连续，不是一次性load 16个float
 * 而是两个寄存器中间间隔步长 例如[data, data+8]，[data+2, data+10]
 */
__attribute__((target("avx512f"))) int Conv1D::conv_8_avx512(float *data, float *res) {
    __m512 data_vec = load_floats_to_m512(data, data + stride_);
    __m512 kernel_vec = kernel_.get_kernel_vec_512()[0];
    __m512 mul = _mm512_mul_ps(data_vec, kernel_vec);
    hadd_m512_halves(&mul,res,res+1);
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
