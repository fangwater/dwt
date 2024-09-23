#include "dwt.hpp"
#include <cstdio>
#include <vector>

int main() {
    std::vector<float> A = {1, 2, 3, 5, 12, 9, 8};
    auto d = DWT("db8",2);
    d.dwt(A.data(), A.size(), 3);
    d.print_coefficients_sum();
    return 0;
}