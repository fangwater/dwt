#include "dwt.hpp"
#include "bior.hpp"
#include "coif.hpp"
#include "conv_1d.hpp"
#include <fmt/format.h>
#include "db.hpp"
#include "sym.hpp"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fmt/format.h>

void DWT::printCoefficients() const {
    // 打印近似系数
    fmt::print("Level {} Approximation Coefficients: \n", detail_coeffs_.size());
    for (const auto &coeff: *approx_coeffs_) {
        fmt::print("{} ", coeff);
    }
    fmt::print("\n");
    for (int level = static_cast<int>(detail_coeffs_.size()) - 1; level >= 0; level--) {
        // 打印细节系数
        fmt::print("Level {} Detail Coefficients: \n", level + 1);
        auto &detail_coeff = detail_coeffs_[level];
        for (const auto &coeff: *detail_coeff) {
            fmt::print("{} ", coeff);
        }
        fmt::print("\n");
    }
}

DWT::DWT(std::string_view filter, int stride = 2) : filter_name_(filter),stride_(stride) {
    window_size_ = [this]() {
        if (filter_name_ == filter::bior5_5::name) {
            kernel_lo_ = const_cast<float *>(filter::bior5_5::low_pass_dec);
            kernel_hi_ = const_cast<float*>(filter::bior5_5::high_pass_dec);
            return filter::bior5_5::window_size;
        } else if (filter_name_ == filter::coif5::name) {
            kernel_lo_ = const_cast<float *>(filter::coif5::low_pass_dec);
            kernel_hi_ = const_cast<float *>(filter::coif5::high_pass_dec);
            return filter::coif5::window_size;
        } else if (filter_name_ == filter::db8::name) {
            kernel_lo_ = const_cast<float *>(filter::db8::low_pass_dec);
            kernel_hi_ = const_cast<float *>(filter::db8::high_pass_dec);
            return filter::db8::window_size;
        } else if (filter_name_ == filter::sym4::name) {
            kernel_lo_ = const_cast<float *>(filter::sym4::low_pass_dec);
            kernel_hi_ = const_cast<float *>(filter::sym4::high_pass_dec);
            return filter::sym4::window_size;
        } else {
            throw std::runtime_error("unsupport filter name");
        }
    }();
    dec_lo_conv_ = std::make_unique<Conv1D>(kernel_lo_, window_size_,stride_);
    dec_hi_conv_ = std::make_unique<Conv1D>(kernel_hi_,window_size_,stride_);
}

std::vector<float> DWT::set_data(const float *data, size_t length) {
    orgignal_length_ = length;
    orginal_signal_ = data;
    right_p_ = [this]() {
        if (orgignal_length_ < window_size_) {
            return window_size_ - orgignal_length_;
        } else {
            return static_cast<size_t>(0);
        }
    }();
    left_p_ = [this]() {
        //floor((N + L - 1)/2)
        res_count_ = std::floor((orgignal_length_ + window_size_ - 1) / 2);
        return res_count_ * stride_;
    }();
    std::vector<float> expand_signal(orgignal_length_ + left_p_ + right_p_);
    pad::symmetric(orginal_signal_, expand_signal.data(), orgignal_length_, left_p_, right_p_);
    return expand_signal;
}

void DWT::dwt(const float *data, size_t length, int level) {
    int max_levels = static_cast<int>(std::floor(std::log2(length)));
    if (level > max_levels) {
        throw std::runtime_error(fmt::format("dwt try using level {} but out of range, max level {}",level,max_levels));
    }
    //level = 1，无需复制数据
    int calc_level = 1;
    while (calc_level <= level) {
        if (calc_level == 1) {
            expand_signal_ = set_data(data, length);
        } else {
            //将上一级得到的低频系数作为输入信号进行卷积
            expand_signal_ = set_data(approx_coeffs_->data(),approx_coeffs_->size());
        }
        //从右向做求卷积
        float *src = expand_signal_.data() + left_p_ - stride_;
        //每次卷积构建一个信号
        auto low_freq_res = std::make_unique<std::vector<float>>(res_count_);
        auto high_freq_res = std::make_unique<std::vector<float>>(res_count_);
        for (size_t i = 0; i < res_count_; i++) {
            dec_lo_conv_->conv_1d(src, low_freq_res->data() + i);
            dec_hi_conv_->conv_1d(src,high_freq_res->data() + i);
            src -= stride_;
        }
        approx_coeffs_ = std::move(low_freq_res);
        detail_coeffs_.push_back(std::move(high_freq_res));
        calc_level++;
    }
}