#include "sym.hpp"
#include "dwt.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

// int main() {
//     std::vector<float> A = {1, 2, 3, 5, 12, 9, 8, 2, 4, 5};
//     auto d = DWT("sym4",2);
//     d.dwt(A.data(), A.size(), 1);
//     d.print_coefficients();
//     return 0;
// }
#include <immintrin.h>
#include <iostream>
#include <vector>


int main() {
    std::vector<float> signal = {1.0f, 2.0f, 3.0f, 4.0f};// 示例信号
    int factor = 2;
    std::vector<float> upsampled(factor * signal.size(), 0.0f);

    // 选择 AVX2 或 AVX512 实现
    upsample_avx2(signal.data(), upsampled.data(), signal.size(), factor);

    // 输出结果
    for (size_t i = 0; i < upsampled.size(); ++i) {
        std::cout << upsampled[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
