//! The project's single sample-rate conversion path.
//!
//! Every consumer downstream of capture — Silero VAD, the native Core ML
//! worker, and the sherpa fallback — wants 16 kHz mono Float32. Core Audio
//! hands us the device rate instead, and the built-in microphone's supported
//! rate list starts at 44.1 kHz (see ADR-0030), so a conversion is unavoidable
//! somewhere. Doing it once, here, is what keeps it off the endpoint path.
//!
//! The kernel is sherpa-onnx's `LinearResampler`, which is Kaldi's
//! `LinearResample`: a Hann-windowed sinc filter with the cutoff at 99% of the
//! Nyquist frequency of the lower rate and `num_zeros = 6`. "Linear" names the
//! linear interpolation of the filter table, not the interpolation of the
//! signal, so this is a bandlimited resampler suitable for ASR input rather
//! than the cheap two-tap kind the name suggests.
//!
//! [`Resampler`] is a streaming object: feed it capture callbacks with
//! [`Resampler::push`] and drain the filter's tail once with
//! [`Resampler::flush`]. Concatenating those outputs is sample-for-sample
//! identical to one [`to_target_rate`] call over the whole signal, which is the
//! property that lets capture do the work incrementally without changing what
//! the model sees.

use std::borrow::Cow;

use anyhow::{anyhow, Result};
use sherpa_onnx::LinearResampler;

/// The rate every ASR backend and the Silero VAD expect.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Streaming converter from one input rate to [`TARGET_SAMPLE_RATE`].
///
/// Holds the filter's carry-over state between calls, so the same instance must
/// see the whole signal in order. Input already at the target rate constructs no
/// filter and passes samples through untouched.
pub struct Resampler {
    /// `None` when the input is already at the target rate.
    ///
    /// The sherpa binding takes `&self` and declares `Sync`, but the underlying
    /// Kaldi object mutates `input_remainder_` and two sample offsets on every
    /// call. Owning it behind `&mut self` here is what actually makes the
    /// aliasing claim true; never hand this type out by shared reference.
    inner: Option<LinearResampler>,
    input_rate: u32,
}

impl Resampler {
    /// Build a converter from `input_rate` to [`TARGET_SAMPLE_RATE`].
    pub fn new(input_rate: u32) -> Result<Self> {
        if input_rate == TARGET_SAMPLE_RATE {
            return Ok(Self {
                inner: None,
                input_rate,
            });
        }
        let input = i32::try_from(input_rate)
            .map_err(|_| anyhow!("input sample rate {input_rate} does not fit in i32"))?;
        if input <= 0 {
            return Err(anyhow!("input sample rate must be positive; got {input_rate}"));
        }
        let target = i32::try_from(TARGET_SAMPLE_RATE)
            .map_err(|_| anyhow!("target sample rate does not fit in i32"))?;
        let inner = LinearResampler::create(input, target).ok_or_else(|| {
            anyhow!("could not build a {input_rate} -> {TARGET_SAMPLE_RATE} Hz resampler")
        })?;
        Ok(Self {
            inner: Some(inner),
            input_rate,
        })
    }

    pub fn input_rate(&self) -> u32 {
        self.input_rate
    }

    /// True when the input already arrives at the target rate and `push`
    /// borrows its argument instead of filtering it.
    pub fn is_pass_through(&self) -> bool {
        self.inner.is_none()
    }

    /// Convert the next contiguous chunk of input.
    ///
    /// Returns fewer output samples than the ratio implies: the filter keeps the
    /// samples it cannot yet center a window on. [`Resampler::flush`] releases
    /// them. An empty return is normal for a short chunk, not a failure.
    pub fn push<'a>(&mut self, chunk: &'a [f32]) -> Cow<'a, [f32]> {
        match &mut self.inner {
            Some(resampler) => Cow::Owned(resampler.resample(chunk, false)),
            None => Cow::Borrowed(chunk),
        }
    }

    /// Drain the filter tail after the last `push`. Empty for pass-through.
    ///
    /// The instance keeps its offsets afterwards, so it is spent: build a new
    /// one for the next signal rather than reusing this.
    pub fn flush(&mut self) -> Vec<f32> {
        match &mut self.inner {
            Some(resampler) => resampler.resample(&[], true),
            None => Vec::new(),
        }
    }
}

/// Convert a complete signal to [`TARGET_SAMPLE_RATE`] in one call.
///
/// Borrows when the input is already at the target rate, so callers on a path
/// that is usually already 16 kHz pay nothing.
pub fn to_target_rate(samples: &[f32], input_rate: u32) -> Result<Cow<'_, [f32]>> {
    if input_rate == TARGET_SAMPLE_RATE {
        return Ok(Cow::Borrowed(samples));
    }
    let mut resampler = Resampler::new(input_rate)?;
    let mut out = resampler.push(samples).into_owned();
    out.extend_from_slice(&resampler.flush());
    Ok(Cow::Owned(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(frequency: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| {
                (std::f32::consts::TAU * frequency * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        (sum / samples.len() as f64).sqrt() as f32
    }

    /// The property the whole incremental-capture design rests on: filtering
    /// the signal in arbitrary callback-sized pieces must produce exactly the
    /// samples one batch call produces. A stateless per-chunk resampler would
    /// fail this at every chunk boundary.
    #[test]
    fn chunked_resampling_equals_one_shot() {
        let signal = tone(440.0, 48_000, 0.5);
        let one_shot = to_target_rate(&signal, 48_000).expect("48k resampler").into_owned();

        // Irregular sizes on purpose: Core Audio callback lengths are not a
        // constant, and 512 is the VAD window so it must not be special.
        for chunking in [&[1_usize, 2, 3, 511, 512, 513, 1024][..], &[97, 4096][..]] {
            let mut resampler = Resampler::new(48_000).expect("48k resampler");
            let mut streamed = Vec::new();
            let mut offset = 0;
            let mut sizes = chunking.iter().copied().cycle();
            while offset < signal.len() {
                let take = sizes.next().unwrap_or(512).min(signal.len() - offset);
                streamed.extend_from_slice(&resampler.push(&signal[offset..offset + take]));
                offset += take;
            }
            streamed.extend_from_slice(&resampler.flush());
            assert_eq!(
                streamed.len(),
                one_shot.len(),
                "chunked output length must match the one-shot length"
            );
            for (index, (a, b)) in streamed.iter().zip(one_shot.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "sample {index} differs: {a} vs {b} for chunking {chunking:?}"
                );
            }
        }
    }

    /// A 12 kHz tone at 48 kHz would fold to 4 kHz without a bandlimiting
    /// filter, landing energy squarely inside the speech band. This is the
    /// test that a naive drop-every-third-sample resampler fails.
    #[test]
    fn out_of_band_content_is_rejected_not_aliased() {
        let signal = tone(12_000.0, 48_000, 0.5);
        let converted = to_target_rate(&signal, 48_000).expect("48k resampler");
        // Skip the filter's ramp-in and ramp-out.
        let interior = &converted[800..converted.len() - 800];
        let attenuation = rms(interior) / rms(&signal);
        assert!(
            attenuation < 0.05,
            "12 kHz stopband content survived at {attenuation} of input RMS \
             (expected below 0.05, i.e. 26 dB of rejection)"
        );
    }

    /// The other half of the filter's job: leave the speech band alone.
    #[test]
    fn in_band_content_keeps_its_amplitude_and_rate() {
        let signal = tone(1_000.0, 48_000, 0.5);
        let converted = to_target_rate(&signal, 48_000).expect("48k resampler");

        let expected_len = signal.len() / 3;
        assert!(
            converted.len().abs_diff(expected_len) <= 1,
            "48k -> 16k must produce one sample per three; got {} for {expected_len}",
            converted.len()
        );

        let interior = &converted[800..converted.len() - 800];
        let ratio = rms(interior) / rms(&signal);
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "1 kHz passband amplitude changed by more than 2%: ratio {ratio}"
        );
    }

    #[test]
    fn target_rate_input_is_passed_through_without_a_filter() {
        let signal = tone(1_000.0, 16_000, 0.1);
        let mut resampler = Resampler::new(16_000).expect("pass-through");
        assert!(resampler.is_pass_through());
        let pushed = resampler.push(&signal);
        assert!(matches!(pushed, Cow::Borrowed(_)), "must not copy");
        assert_eq!(pushed.as_ref(), signal.as_slice());
        assert!(resampler.flush().is_empty());

        let converted = to_target_rate(&signal, 16_000).expect("pass-through");
        assert!(matches!(converted, Cow::Borrowed(_)));
        assert_eq!(converted.as_ref(), signal.as_slice());
    }

    /// 44.1 kHz is the other rate the built-in microphone advertises, and its
    /// ratio to 16 kHz is not an integer, so the phase table has many more
    /// entries than the 48 kHz case.
    #[test]
    fn non_integer_ratio_resamples_at_the_right_length() {
        let signal = tone(1_000.0, 44_100, 0.5);
        let converted = to_target_rate(&signal, 44_100).expect("44.1k resampler");
        let expected_len = signal.len() * 16_000 / 44_100;
        assert!(
            converted.len().abs_diff(expected_len) <= 1,
            "44.1k -> 16k produced {} samples, expected about {expected_len}",
            converted.len()
        );
        let interior = &converted[800..converted.len() - 800];
        let ratio = rms(interior) / rms(&signal);
        assert!(
            (ratio - 1.0).abs() < 0.02,
            "1 kHz passband amplitude changed by more than 2% at 44.1 kHz: ratio {ratio}"
        );
    }

    #[test]
    fn zero_input_rate_is_rejected() {
        assert!(Resampler::new(0).is_err());
    }
}
