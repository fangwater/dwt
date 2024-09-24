#ifndef IDWT_HPP
#define IDWT_HPP
#include <cstddef>
#include <cstring>
#include <immintrin.h>
class IDWT {
public:
    static void upsample(const float *signal, float *upsampled, size_t length, int factor) __attribute__((target("default")));
    static void upsample(const float *signal, float *upsampled, size_t length, int factor) __attribute__((target("avx512f")));
};

#endif
