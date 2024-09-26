#ifndef IDWT_HPP
#define IDWT_HPP
#include "conv_1d.hpp"
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <string_view>
#include <vector>
#include <string>
class IDWT {
public:
    IDWT() = delete;
    IDWT(std::string_view filter, int stride);
    void idwt(const std::unique_ptr<std::vector<float>> &approx_coeff, std::vector<std::unique_ptr<std::vector<float>>> &detail_coeffs);
    static void upsample(const float *signal, float *upsampled, size_t length, int factor) __attribute__((target("default")));
    static void upsample(const float *signal, float *upsampled, size_t length, int factor) __attribute__((target("avx512f")));
public:
    std::string filter_name_;
    int window_size_;
    int stride_;
    std::unique_ptr<std::vector<float>> reconstructed_signal_;
private:
    std::unique_ptr<Conv1D> rec_lo_conv_;
    std::unique_ptr<Conv1D> rec_hi_conv_;
    
};

#endif
