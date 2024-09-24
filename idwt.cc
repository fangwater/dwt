#include "idwt.hpp"

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