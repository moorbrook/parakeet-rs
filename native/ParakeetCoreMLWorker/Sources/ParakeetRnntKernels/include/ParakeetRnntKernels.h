#ifndef PARAKEET_RNNT_KERNELS_H
#define PARAKEET_RNNT_KERNELS_H

#include <stddef.h>

/// The two inner loops of the native RNNT decode loop.
///
/// They live in C because Swift will not vectorize them. `SIMD8<Float>` built
/// from a `SIMD8<Float16>` compiles to an outlined runtime call with a register
/// spill around it — measured at about 3 GB/s against the 13.1 MB a prediction
/// step reads, some thirty times slower than the same loop written here.
/// Everything else in the decode loop stays in Swift.

#ifdef __cplusplus
extern "C" {
#endif

/// `out[r] = fp16(bias[r] + row(r) . vector)` for `r` in `[rowBegin, rowEnd)`.
///
/// `matrix` is row-major fp16 with `columns` values per row, `vector` and
/// `bias` are fp32. Products accumulate in fp32 and the result is rounded to
/// fp16, which is what the compiled model's tensors are. Row slices are
/// independent, so splitting the range across threads cannot change a result.
void parakeet_rnnt_matvec(
    const unsigned short *matrix,
    const float *vector,
    const float *bias,
    float *out,
    size_t columns,
    size_t rowBegin,
    size_t rowEnd);

/// Round `count` fp32 values in place to the nearest fp16, as a cast to an
/// fp16 tensor does.
void parakeet_rnnt_round_fp16(float *values, size_t count);

#ifdef __cplusplus
}
#endif

#endif
