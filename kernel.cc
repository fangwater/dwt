#include "kernel.hpp"
#include <cstring>
// 检查编译器是否支持 AVX 和 AVX-512
constexpr bool has_avx_support() {
#if defined(__AVX__)
    return true;
#else
    return false;
#endif
}

constexpr bool has_avx512_support() {
#if defined(__AVX512F__)
    return true;
#else
    return false;
#endif
}


__attribute__((target("default"))) void Kernel::initialize(const float *kernel_data) {
    std::cout << "default initialize" << std::endl;
    int chunks256 = kernel_size_ / 8;
    int remainder = kernel_size_ % 8;
    __m256 *kernel_vec256 = nullptr;

    // Allocate memory for the chunks, adding one more if there's a remainder
    if (remainder > 0) {
        kernel_vec256 = new __m256[chunks256 + 1];
    } else {
        kernel_vec256 = new __m256[chunks256];
    }

    // Load each full chunk
    for (int i = 0; i < chunks256; ++i) {
        kernel_vec256[i] = _mm256_loadu_ps(kernel_data + i * 8);
    }

    // Handle the remainder by zero-padding
    if (remainder > 0) {
        float temp[8] = {0};
        std::memcpy(temp, kernel_data + chunks256 * 8, remainder * sizeof(float));
        kernel_vec256[chunks256] = _mm256_loadu_ps(temp);
    }

    // Assign the pointer to the kernel
    kernel_ = reinterpret_cast<void *>(kernel_vec256);
}


__attribute__((target("avx512f"))) void Kernel::initialize(const float *kernel_data) {
    std::cout << "avx512f initialize" << std::endl;
    int chunks512 = kernel_size_ / 16;
    int remainder = kernel_size_ % 16;
    __m512 *kernel_vec512 = nullptr;

    // Allocate memory for the chunks, adding one more if there's a remainder
    if (remainder > 0) {
        kernel_vec512 = new __m512[chunks512 + 1];
    } else {
        kernel_vec512 = new __m512[chunks512];
    }

    // Load each full chunk
    for (int i = 0; i < chunks512; ++i) {
        kernel_vec512[i] = _mm512_loadu_ps(kernel_data + i * 16);
    }

    // Handle the remainder by zero-padding
    if (remainder > 0) {
        float temp[16] = {0};
        std::memcpy(temp, kernel_data + chunks512 * 16, remainder * sizeof(float));
        kernel_vec512[chunks512] = _mm512_loadu_ps(temp);
    }

    // Assign the pointer to the kernel
    kernel_ = reinterpret_cast<void *>(kernel_vec512);
}

Kernel::Kernel(const float *kernel_data, int kernel_size)
    : kernel_size_(kernel_size) {
    initialize(kernel_data);
}


int Kernel::get_kernel_size() const {
    return kernel_size_;
}
