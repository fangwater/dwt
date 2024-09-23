#include <cpuid.h>
#include <iostream>

// 检查 AVX 支持
bool supports_avx() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid(1, eax, ebx, ecx, edx);
    return (ecx & bit_AVX) != 0;
}

// 检查 AVX512F 支持
bool supports_avx512f() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & bit_AVX512F) != 0;
}


int main() {
    bool avx = supports_avx();
    bool avx512f = supports_avx512f();

    std::cout << "AVX: " << (avx ? "Supported" : "Not supported") << std::endl;
    std::cout << "AVX512F: " << (avx512f ? "Supported" : "Not supported") << std::endl;
    std::cout << ((avx && avx512f) ? 1 : 0) << std::endl;
    return (avx && avx512f) ? 1 : 0;
}
