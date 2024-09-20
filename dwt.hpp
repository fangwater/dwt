#ifndef DWT_HPP
#define DWT_HPP
#include <cstddef>
#include <immintrin.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "conv_1d.hpp"
namespace pad {
    template<typename T>
    void symmetric(const T *A, T *B, size_t L, size_t N) {
        size_t period = 2 * L - 2;
        for (size_t i = 0; i < N; ++i) {
            size_t idx = i % period;
            if (idx >= L) {
                idx = period - idx;
            }
            B[i] = A[idx];
        }
    }
    template<typename T>
    std::vector<T> symmetric(const std::vector<T> &A, size_t N) {
        std::vector<T> B(N);
        symmetric(A.data(), B.data(), A.size(), B.size());
        return B;
    }
}// namespace pad


namespace pad {
    template<typename T>
    void symmetric(const T *A, T *B, size_t L, size_t left, size_t right) {
        size_t N = L + left + right;
        size_t period = 2 * L;
        for (size_t i = 0; i < N; ++i) {
            ptrdiff_t k = static_cast<ptrdiff_t>(i) - static_cast<ptrdiff_t>(left);
            ptrdiff_t idx = ((k % static_cast<ptrdiff_t>(period)) + static_cast<ptrdiff_t>(period)) % static_cast<ptrdiff_t>(period);
            if (idx >= static_cast<ptrdiff_t>(L)) {
                idx = 2 * static_cast<ptrdiff_t>(L) - idx - 1;
            }
            B[i] = A[idx];
        }
    }

    template<typename T>
    std::vector<T> symmetric(const std::vector<T> &A, size_t left, size_t right) {
        size_t L = A.size();
        size_t N = L + left + right;
        std::vector<T> B(N);
        size_t period = 2 * L;
        for (size_t i = 0; i < N; ++i) {
            ptrdiff_t k = static_cast<ptrdiff_t>(i) - static_cast<ptrdiff_t>(left);
            ptrdiff_t idx = ((k % static_cast<ptrdiff_t>(period)) + static_cast<ptrdiff_t>(period)) % static_cast<ptrdiff_t>(period);
            if (idx >= static_cast<ptrdiff_t>(L)) {
                idx = 2 * static_cast<ptrdiff_t>(L) - idx - 1;
            }
            B[i] = A[idx];
        }
        return B;
    }
}// namespace pad

class DWT {
public:
    DWT(std::string_view filter, int stride);
    DWT() = delete;
    void set_data(float *data, size_t length);
    std::vector<float> &dwt();

public:
    std::string filter_name_;
    int window_size_;
    float *kernel_lo_;
    float *kernel_hi_;
    float *orginal_signal_;
    size_t orgignal_length_;
    size_t left_p_;
    size_t right_p_;
    int stride_;
    std::vector<float> result_;
    std::vector<float> expand_signal_;

private:
    std::unique_ptr<Conv1D> dec_lo_conv_;
    std::unique_ptr<Conv1D> dec_hi_conv_;
};


#endif