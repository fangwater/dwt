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

__attribute__((target("default"))) int Conv1D::conv_12(float *data, float *res) {
    // Load first 8 elements of data
    __m256 data_vec1 = _mm256_loadu_ps(data);
    // Load next 4 elements of data into lower half of __m256, upper half is zero
    __m128 data_high = _mm_loadu_ps(data + 8);
    __m256 data_vec2 = _mm256_castps128_ps256(data_high);

    // Load first 8 elements of kernel
    __m256 kernel_vec1 = kernel_.get_kernel_vec_256()[0];
    // Load next 4 elements of kernel into lower half of __m256, upper half is zero
    __m128 kernel_high = _mm_loadu_ps(reinterpret_cast<float *>(kernel_.get_kernel_vec_256() + 1));
    __m256 kernel_vec2 = _mm256_castps128_ps256(kernel_high);

    // Perform element-wise multiplication
    __m256 mul1 = _mm256_mul_ps(data_vec1, kernel_vec1);
    __m256 mul2 = _mm256_mul_ps(data_vec2, kernel_vec2);

    // Sum the results horizontally
    float sum1, sum2;
    horizontal_sum_m256(&mul1,&sum1);
    horizontal_sum_m256(&mul2,&sum2);

    // Store the final result
    res[0] = sum1 + sum2;
    return 1;
}

__attribute__((target("avx512f"))) int Conv1D::conv_12(float *data, float *res) {
    __mmask16 mask = 0x0FFF;// Lower 12 bits set to 1
    __m512 data_vec = _mm512_maskz_loadu_ps(mask, data);
    __m512 kernel_vec = _mm512_maskz_loadu_ps(mask, reinterpret_cast<float *>(kernel_.get_kernel_vec_512()));
    __m512 mul = _mm512_mul_ps(data_vec, kernel_vec);
    float sum = _mm512_reduce_add_ps(mul);
    res[0] = sum;
    return 1;
}

__attribute__((target("default"))) int Conv1D::conv_16(float *data, float *res) {
    __m256 data_vec1 = _mm256_loadu_ps(data);
    __m256 data_vec2 = _mm256_loadu_ps(data + 8);
    __m256 kernel_vec1 = kernel_.get_kernel_vec_256()[0];
    __m256 kernel_vec2 = kernel_.get_kernel_vec_256()[1];
    __m256 mul1 = _mm256_mul_ps(data_vec1, kernel_vec1);
    __m256 mul2 = _mm256_mul_ps(data_vec2, kernel_vec2);
    float sum1, sum2;
    horizontal_sum_m256(&mul1, &sum1);
    horizontal_sum_m256(&mul2, &sum2);
    res[0] = sum1 + sum2;

    return 1;
}

__attribute__((target("avx512f"))) int Conv1D::conv_16(float *data, float *res) {
    __m512 data_vec = _mm512_loadu_ps(data);
    __m512 kernel_vec = kernel_.get_kernel_vec_512()[0];
    __m512 mul = _mm512_mul_ps(data_vec, kernel_vec);
    float sum = _mm512_reduce_add_ps(mul);
    res[0] = sum;
    return 1;
}

__attribute__((target("default"))) int Conv1D::conv_30(float *data, float *res) {
    float result = 0.0f;

    for (int i = 0; i < 3; i++) {
        __m256 data_vec = _mm256_loadu_ps(data);
        __m256 kernel_vec = kernel_.get_kernel_vec_256()[i];
        __m256 mul = _mm256_mul_ps(data_vec, kernel_vec);
        float part_sum;
        horizontal_sum_m256(&mul, &part_sum);
        result += part_sum;
    }
    {
        alignas(32) int32_t mask_values[8] = {-1, -1, -1, -1, -1, -1, 0, 0};
        __m256i mask = _mm256_load_si256(reinterpret_cast<const __m256i *>(mask_values));

        __m256 data_vec = _mm256_maskload_ps(data + 24, mask);
        __m256 kernel_vec = _mm256_maskload_ps(reinterpret_cast<float *>(kernel_.get_kernel_vec_256()) + 24, mask);
        __m256 mul = _mm256_mul_ps(data_vec, kernel_vec);
        float part_sum;
        horizontal_sum_m256(&mul, &part_sum);
        result += part_sum;
    }
    res[0] = result;
    return 1;
}

__attribute__((target("avx512f"))) int Conv1D::conv_30(float *data, float *res) {
    float result = 0.0f;
    {
        __m512 data_vec = _mm512_loadu_ps(data);
        __m512 kernel_vec = kernel_.get_kernel_vec_512()[0];
        __m512 mul = _mm512_mul_ps(data_vec, kernel_vec);
        result += _mm512_reduce_add_ps(mul);
    }

    {
        __mmask16 mask = 0x3FFF;
        __m512 data_vec = _mm512_maskz_loadu_ps(mask, data + 16);
        __m512 kernel_vec = _mm512_maskz_loadu_ps(mask, reinterpret_cast<float *>(kernel_.get_kernel_vec_512()) + 16);
        __m512 mul = _mm512_mul_ps(data_vec, kernel_vec);
        result += _mm512_reduce_add_ps(mul);
    }

    res[0] = result;
    return 1;
}