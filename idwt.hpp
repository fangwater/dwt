#ifndef IDWT_HPP
#define IDWT_HPP
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <string_view>
#include <vector>
class IDWT {
public:
    IDWT() = delete;
    IDWT(std::string_view filter, int stride);
    void idwt(const std::unique_ptr<std::vector<float>>& approx_coeff, std::vector<std::unique_ptr<std::vector<float>>>& detail_coeffs);

public:
    static void upsample(const float *signal, float *upsampled, size_t length, int factor) __attribute__((target("default")));
    static void upsample(const float *signal, float *upsampled, size_t length, int factor) __attribute__((target("avx512f")));
};

#endif
