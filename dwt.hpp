#ifndef DWT_HPP
#define DWT_HPP
#include <cstddef>
#include <vector>
#include <string_view>
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


std::vector<float> signal_expand(std::string_view filter_name, float *siganl, size_t length);
#endif