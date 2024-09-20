#ifndef KERNEL_HPP
#define KERNEL_HPP
#include <immintrin.h>
constexpr bool has_avx512_support();
constexpr bool has_avx_support();

class Kernel {
public:
    Kernel(const float *kernel_data, int kernel_size);

    // 获取预加载的内核数据
    void* get_kernel_vec() const;
    // 获取内核大小
    int get_kernel_size() const;

private:
    void initialize(const float *kernel_data) __attribute__((target("default")));
    void initialize(const float *kernel_data) __attribute__((target("avx512f")));
    __m512 *get_kernel_vec() __attribute__((target("avx512f"))) {
        return reinterpret_cast<__m512*>(kernel_);
    }
    int kernel_size_;
    void* kernel_;
};

#endif// KERNEL_HPP
