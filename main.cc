#include "db.hpp"
#include "dwt.hpp"
#include "conv_1d.hpp"
#include "kernel.hpp"
#include <cstdio>
#include <vector>

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
}
