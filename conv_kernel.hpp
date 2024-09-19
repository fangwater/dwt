#ifndef CONV_KERNEL
#define CONV_KERNEL
/**
 * @brief 根据不同的，产生不同的卷积核
 */

#include <cstddef>
void Conv1D(int kernel_size);


//8,12,16,30
void conv_8_simd256(float *data, float *kernel, float *res);

// 卷积核大小为12的卷积操作，步长为2
void conv_12_simd256(float *data, size_t length, float *res);

// 卷积核大小为16的卷积操作，步长为2
void conv_16_simd256(float *data, size_t length, float *res);

// 卷积核大小为30的卷积操作，步长为2
void conv_30_simd256(float *data, size_t length, float *res);


#endif