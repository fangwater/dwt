#include "idwt.hpp"
#include "bior.hpp"
#include "coif.hpp"
#include "conv_1d.hpp"
#include "db.hpp"
#include "sym.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

IDWT::IDWT(std::string_view filter, int stride) : filter_name_(filter), stride_(stride), reconstructed_signal_(nullptr){
    float *kernel_lo, *kernel_hi;
    window_size_ = [&]() {
        if (filter_name_ == filter::bior5_5::name) {
            kernel_lo = const_cast<float *>(filter::bior5_5::low_pass_rec);
            kernel_hi = const_cast<float *>(filter::bior5_5::high_pass_rec);
            return filter::bior5_5::window_size;
        } else if (filter_name_ == filter::coif5::name) {
            kernel_lo = const_cast<float *>(filter::coif5::low_pass_rec);
            kernel_hi = const_cast<float *>(filter::coif5::high_pass_rec);
            return filter::coif5::window_size;
        } else if (filter_name_ == filter::db8::name) {
            kernel_lo = const_cast<float *>(filter::db8::low_pass_rec);
            kernel_hi = const_cast<float *>(filter::db8::high_pass_rec);
            return filter::db8::window_size;
        } else if (filter_name_ == filter::sym4::name) {
            kernel_lo = const_cast<float *>(filter::sym4::low_pass_rec);
            kernel_hi = const_cast<float *>(filter::sym4::high_pass_rec);
            return filter::sym4::window_size;
        } else {
            throw std::runtime_error("unsupport filter name");
        }
    }();
    std::vector<float> reverse_kernel_lo(kernel_lo, kernel_lo + window_size_);
    std::vector<float> reverse_kernel_hi(kernel_hi, kernel_hi + window_size_);
    std::reverse(reverse_kernel_lo.begin(), reverse_kernel_lo.end());
    std::reverse(reverse_kernel_hi.begin(), reverse_kernel_hi.end());
    rec_lo_conv_ = std::make_unique<Conv1D>(reverse_kernel_lo.data(), window_size_, stride_);
    rec_hi_conv_ = std::make_unique<Conv1D>(reverse_kernel_hi.data(), window_size_, stride_);
}


__attribute__((target("default"))) void IDWT::upsample(const float *signal, float *upsampled, size_t length, int factor) {
    size_t i = 0;
    std::memset(upsampled, 0, factor * length * sizeof(float));
    for (; i + 8 <= length; i += 8) {
        __m256 vec = _mm256_loadu_ps(&signal[i]);
        _mm256_storeu_ps(&upsampled[i * factor + 1], vec);
    }
    for (; i < length; ++i) {
        upsampled[i * factor + 1] = signal[i];
    }
}
__attribute__((target("avx512f"))) void IDWT::upsample(const float *signal, float *upsampled, size_t length, int factor) {
    size_t i = 0;
    std::memset(upsampled, 0, factor * length * sizeof(float));
    for (; i + 16 <= length; i += 16) {
        __m512 vec = _mm512_loadu_ps(&signal[i]);
        _mm512_storeu_ps(&upsampled[i * factor + 1], vec);
    }
    for (; i < length; ++i) {
        upsampled[i * factor + 1] = signal[i];
    }
}


void IDWT::idwt(const std::unique_ptr<std::vector<float>> &approx_coeff, std::vector<std::unique_ptr<std::vector<float>>> &detail_coeffs) {
    auto& current_signal = approx_coeff;
    //从最高层开始逐层重构
    for (auto &detail_coeff: detail_coeffs) {
        auto upsampled_approx = std::make_unique<std::vector<float>>(current_signal->size() * 2);
        auto upsampled_detail = std::make_unique<std::vector<float>>(detail_coeff->size() * 2);
        upsample(current_signal->data(), upsampled_approx->data(), current_signal->size(), 2);
        upsample(detail_coeff->data(), upsampled_detail->data(), detail_coeff->size(), 2);
        //分别与低通滤波器和高通滤波器进行valid卷积
        auto N = upsampled_approx->size() - window_size_ + 1;
        auto recon_approx = std::make_unique<std::vector<float>>(N);
        auto recon_detail = std::make_unique<std::vector<float>>(N);
        for (int i = 0; i < N; i++) {
            rec_lo_conv_->conv_1d(upsampled_approx->data() + i, recon_approx->data() + i);
            rec_hi_conv_->conv_1d(upsampled_detail->data() + i, recon_detail->data() + i);
        }
        reconstructed_signal_ = std::make_unique<std::vector<float>>(recon_approx->size());
        for (int i = 0; i < reconstructed_signal_->size(); i++) {
            reconstructed_signal_->at(i) = recon_approx->at(i) + recon_detail->at(i);
        }
        print_vector(*reconstructed_signal_);
    }
}