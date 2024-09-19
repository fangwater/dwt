#include "dwt.hpp"
#include "bior.hpp"
#include "coif.hpp"
#include "conv_kernel.hpp"
#include "db.hpp"
#include "sym.hpp"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

DWT::DWT(std::string_view filter, int stride) : filter_name_(filter),stride_(stride) {
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
}

void DWT::set_data(float *data, size_t length) {
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
        size_t res_count = std::floor((orgignal_length_ + window_size_ - 1) / 2);
        result_.resize(res_count);
        return res_count * stride_;
    }();
}

std::vector<float> &DWT::dwt() {
    expand_signal_.resize(orgignal_length_ + left_p_ + right_p_);
    pad::symmetric(orginal_signal_, expand_signal_.data(), orgignal_length_, left_p_, right_p_);
    //从右向做求卷积
    float *s = expand_signal_.data() + left_p_ - stride_;
    for (int i = 0; i < result_.size(); i++) {
        float ss = 0;
        if (window_size_ == 8) {
            conv_8_simd256(s, kernel_lo_,&result_[i]);
        } else {
            for (int k = 0; k < window_size_; k++) {
                ss += (*(s + k) * kernel_lo_[k]);
            }
            result_[i] = ss;
        }
        s = s - stride_;
    }
    return result_;
}