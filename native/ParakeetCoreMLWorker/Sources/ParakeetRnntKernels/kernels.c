#include "include/ParakeetRnntKernels.h"

#include <arm_neon.h>

void parakeet_rnnt_matvec(
    const unsigned short *matrix,
    const float *vector,
    const float *bias,
    float *out,
    size_t columns,
    size_t rowBegin,
    size_t rowEnd)
{
    const __fp16 *weights = (const __fp16 *)matrix;
    for (size_t row = rowBegin; row < rowEnd; row++) {
        const __fp16 *w = weights + row * columns;
        // Four accumulators so the FMA latency chain is covered; they are
        // summed in a fixed order, so the result does not depend on how the
        // rows were split across threads.
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        size_t column = 0;
        for (; column + 16 <= columns; column += 16) {
            float16x8_t h0 = vld1q_f16(w + column);
            float16x8_t h1 = vld1q_f16(w + column + 8);
            a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(vector + column));
            a1 = vfmaq_f32(a1, vcvt_high_f32_f16(h0), vld1q_f32(vector + column + 4));
            a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(vector + column + 8));
            a3 = vfmaq_f32(a3, vcvt_high_f32_f16(h1), vld1q_f32(vector + column + 12));
        }
        float total = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        for (; column < columns; column++) {
            total += (float)w[column] * vector[column];
        }
        out[row] = (float)(__fp16)(total + bias[row]);
    }
}

void parakeet_rnnt_round_fp16(float *values, size_t count)
{
    size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        float32x4_t low = vld1q_f32(values + index);
        float32x4_t high = vld1q_f32(values + index + 4);
        float16x8_t rounded = vcvt_high_f16_f32(vcvt_f16_f32(low), high);
        vst1q_f32(values + index, vcvt_f32_f16(vget_low_f16(rounded)));
        vst1q_f32(values + index + 4, vcvt_high_f32_f16(rounded));
    }
    for (; index < count; index++) {
        values[index] = (float)(__fp16)values[index];
    }
}
