#include "conv_1d.hpp"
#include "dwt.hpp"
#include "fmt/core.h"
#include "idwt.hpp"
#include <vector>
void test() {
    std::vector<float> input_signal{1, 2, 3, 5, 12, 9, 8, 2, 4, 5};
    auto dwt = DWT("bior5.5", 2);
    dwt.dwt(input_signal.data(), input_signal.size(), 1);
    fmt::print("approx_coeff:\n");
    print_vector(*dwt.approx_coeff_);
    auto idwt = IDWT("bior5.5", 2);
    idwt.idwt(dwt.approx_coeff_,dwt.detail_coeffs_);
}

int main() {
    test();
    return 0;
};