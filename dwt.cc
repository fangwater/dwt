#include "dwt.hpp"
#include "bior.hpp"
#include "coif.hpp"
#include "db.hpp"
#include "sym.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>

std::vector<float> signal_expand(std::string_view filter_name, float *siganl, size_t length) {
    size_t window_size;
    if (filter_name == filter::bior5_5::name) {
        window_size = filter::bior5_5::window_size;
    } else if (filter_name == filter::coif5::name) {
        window_size = filter::coif5::window_size;
    } else if (filter_name == filter::db8::name) {
        window_size = filter::db8::window_size;
    } else if (filter_name == filter::sym4::name) {
        window_size = filter::sym4::window_size;
    } else {
        throw std::runtime_error("unsupport filter name");
    }
    //floor((N + L - 1)/2)
    size_t signal_count = std::floor((length + window_size - 1) / 2);
    /**
     * @brief dwt卷积需要降采样，padding_w = np.ceil(((signal_count * 2 + len(window) - 1 ) - len(signal))/2)
     */
    int padding_width = std::ceil(((signal_count * 2 + window_size - 1) - length) / 2);
    std::vector<float> res(length + 2 * padding_width);
    pad::symmetric(siganl, res.data(), length, res.size());
    return res;
}