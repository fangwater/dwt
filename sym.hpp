#ifndef SYM_FILTER_HPP
#define SYM_FILTER_HPP
#include <string_view>
namespace filter {
    namespace sym4 {
        constexpr std::string_view name = "sym4";
        const int window_size = 8;
        const float high_pass_dec[] = {
                -0.0322231006040427,
                -0.012603967262037833,
                0.09921954357684722,
                0.29785779560527736,
                -0.8037387518059161,
                0.49761866763201545,
                0.02963552764599851,
                -0.07576571478927333};
        const float low_pass_dec[] = {
                -0.07576571478927333,
                 -0.02963552764599851,
                 0.49761866763201545,
                 0.8037387518059161,
                 0.29785779560527736,
                 -0.09921954357684722,
                 -0.012603967262037833,
                 0.0322231006040427};
    }
}// namespace filter

#endif