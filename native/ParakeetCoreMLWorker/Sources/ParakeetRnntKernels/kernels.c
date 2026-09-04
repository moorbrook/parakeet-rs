#include "include/ParakeetRnntKernels.h"

#include <arm_neon.h>
#include <stdatomic.h>

static atomic_int parakeet_rnnt_recording_flag = 0;

void parakeet_rnnt_set_recording(int recording)
{
    atomic_store_explicit(&parakeet_rnnt_recording_flag, recording, memory_order_relaxed);
}

int parakeet_rnnt_recording(void)
{
    return atomic_load_explicit(&parakeet_rnnt_recording_flag, memory_order_relaxed);
}

/// Largest vector the fp16 fast path converts on the stack. Every matrix in the
/// decode loop is narrower than this (1280 for the interleaved LSTM gates, 640
/// for the joint projections); anything wider falls back to converting each
/// weight instead.
#define PARAKEET_MAX_COLUMNS 2048

__attribute__((target("arch=armv8.4-a+fp16fml")))
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

    // The vector is narrowed once per call, which loses nothing: every vector
    // the decode loop passes has already crossed an fp16 tensor boundary (an
    // embedding row, a rounded LSTM hidden state, a rounded and rectified joint
    // sum). Narrowing it lets the product run as FMLAL, which widens both
    // operands inside the multiply and halves the instruction count against
    // converting the weights and multiplying in fp32. Both are fused, with the
    // same lane pairing and the same four accumulators, so they agree exactly.
    __fp16 narrowed[PARAKEET_MAX_COLUMNS];
    const int narrow = columns <= PARAKEET_MAX_COLUMNS;
    if (narrow) {
        size_t index = 0;
        for (; index + 8 <= columns; index += 8) {
            vst1q_f16(
                narrowed + index,
                vcvt_high_f16_f32(
                    vcvt_f16_f32(vld1q_f32(vector + index)), vld1q_f32(vector + index + 4)));
        }
        for (; index < columns; index++) {
            narrowed[index] = (__fp16)vector[index];
        }
    }

    for (size_t row = rowBegin; row < rowEnd; row++) {
        const __fp16 *w = weights + row * columns;
        // Four accumulators so the multiply-add latency chain is covered; they
        // are summed in a fixed order, so the result does not depend on how the
        // rows were split across threads.
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        size_t column = 0;
        if (narrow) {
            for (; column + 16 <= columns; column += 16) {
                float16x8_t w0 = vld1q_f16(w + column);
                float16x8_t w1 = vld1q_f16(w + column + 8);
                float16x8_t x0 = vld1q_f16(narrowed + column);
                float16x8_t x1 = vld1q_f16(narrowed + column + 8);
                a0 = vfmlalq_low_f16(a0, w0, x0);
                a1 = vfmlalq_high_f16(a1, w0, x0);
                a2 = vfmlalq_low_f16(a2, w1, x1);
                a3 = vfmlalq_high_f16(a3, w1, x1);
            }
        } else {
            for (; column + 16 <= columns; column += 16) {
                float16x8_t h0 = vld1q_f16(w + column);
                float16x8_t h1 = vld1q_f16(w + column + 8);
                a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(vector + column));
                a1 = vfmaq_f32(a1, vcvt_high_f32_f16(h0), vld1q_f32(vector + column + 4));
                a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(vector + column + 8));
                a3 = vfmaq_f32(a3, vcvt_high_f32_f16(h1), vld1q_f32(vector + column + 12));
            }
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
