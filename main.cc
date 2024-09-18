#include "db.hpp"
#include "dwt.hpp"
#include "conv_kernel.hpp"
#include <cinttypes>
#include <cstdio>
#include <vector>
void conv(float *data, size_t length, float *kernel, int window_size, float *res) {
    size_t res_index = 0;

    // 进行卷积操作，步长为2
    for (size_t i = 0; i + window_size <= length; i += 2) {
        float sum = 0.0f;
        // 累加卷积核与数据的乘积
        for (size_t j = 0; j < window_size; ++j) {
            sum += data[i + j] * kernel[j];
        }
        // 将结果保存到输出数组
        res[res_index++] = sum;
    }
}
int main() {
    std::vector<float> A = {2, 3, 4, 5, 12, 9, 8};
    auto v = signal_expand("db8", A.data(), A.size());
    auto res = std::vector<float>(11);
    conv(v.data(), v.size(), const_cast<float *>(filter::db8::low_pass_dec), filter::db8::window_size, res.data());
    for (auto &r: res) {
        printf("%.3f\n",r);
    }
    return 0;
}
