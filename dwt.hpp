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
private:
    std::vector<float> set_data(const float *data, size_t length);

public:
    DWT(std::string_view filter, int stride);
    DWT() = delete;
    void dwt(const float *data, size_t length, int level);
    void print_coefficients() const;
    void print_coefficients_sum() const;
public:
    std::string filter_name_;
    int window_size_;
    float *kernel_lo_;
    float *kernel_hi_;
    const float *orginal_signal_;
    size_t orgignal_length_;
    size_t left_p_;
    size_t right_p_;
    size_t res_count_;
    int stride_;
    std::vector<float> expand_signal_;
    // -------
    // [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
    //     Ordered list of coefficients arrays
    //     where ``n`` denotes the level of decomposition. The first element
    //     (``cA_n``) of the result is approximation coefficients array and the
    //     following elements (``cD_n`` - ``cD_1``) are details coefficients
    //     arrays.
    // --------
    /**
     * @brief 根据pywt的实现，近似系数（低频部分）只保留最后一层
     * 近似系数表示信号的低频部分，经过多层变换后，只有最后一层的低频信息最为重要
     * 在多层小波变换中，每一层的近似系数都会逐层向下传递，最终保留的是最高层的近似系数
     */
    std::unique_ptr<std::vector<float>> approx_coeff_;
    //每一层的细节系数（高频部分）,每一层都保存
    std::vector<std::unique_ptr<std::vector<float>>> detail_coeffs_;

private:
    std::unique_ptr<Conv1D> dec_lo_conv_;
    std::unique_ptr<Conv1D> dec_hi_conv_;
};


#endif