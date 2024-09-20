#include "db.hpp"
#include "dwt.hpp"
#include "conv_kernel.hpp"
#include "kernel.hpp"
#include <cstdio>
#include <vector>
void conv(float *data, size_t length, float *kernel, int window_size, int step, float *res) {
    size_t res_index = 0;
    for (size_t i = step; i + window_size <= length; i += step) {
        float sum = 0.0f;
        for (size_t j = 0; j < window_size; ++j) {
            sum += data[i + j] * kernel[window_size - 1 - j];
            printf("%.1f ",data[i+j]);
        }
        res[res_index++] = sum;
    }
}

// int main() {
//     std::vector<float> A = {1, 2, 3, 5, 12, 9, 8};
//     auto d = DWT("sym4", 2);
//     d.set_data(A.data(), A.size());
//     auto res = d.dwt();
//     for (int i = 0; i < res.size(); i++) {
//         printf("%.3f\n ", res[i]);
//     }
//     printf("\n");
//     return 0;
// }

int main() {
    std::vector<float> A = {1, 2, 3, 5, 12, 9, 8,0,1};
    Kernel k(A.data(), 9);
    auto p1 = k.get_kernel_vec_256();
    printf("get 256\n");
    auto p2 = k.get_kernel_vec_512();
    printf("get 512\n");
}
