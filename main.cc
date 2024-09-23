#include "dwt.hpp"
#include <cstdio>
#include <vector>

int main() {
    std::vector<float> A = {1, 2, 3, 5, 12, 9, 8};
    auto d = DWT("sym4", 2);
    d.dwt(A.data(), A.size(), 2);
    d.printCoefficients();
    return 0;
}