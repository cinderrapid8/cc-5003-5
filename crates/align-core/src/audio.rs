use crate::error::{AlignError, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

/// Interleaved multi-channel audio. All channels share one nominal sample rate.
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved frames: data[frame * channels + ch].
    pub data: Vec<f32>,
}

impl AudioBuffer {
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels,
            data: Vec::new(),
        }
    }

    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.data.len() / self.channels as usize
        }
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }

    /// Mono down-mix used for analysis only. The original per-channel data is kept.
    pub fn mono_mix(&self) -> Vec<f32> {
        let ch = self.channels as usize;
        self.data
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    }

    pub fn channel(&self, c: usize) -> Vec<f32> {
        let ch = self.channels as usize;
        let n = self.frames();
        (0..n).map(|f| self.data[f * ch + c]).collect()
    }

    /// Sample value at fractional frame index via linear interpolation,
    /// clamped to the recorded range.
    pub fn sample_linear(&self, c: usize, frame: f64) -> f32 {
        let n = self.frames();
        if n == 0 {
            return 0.0;
        }
        if frame <= 0.0 {
            return self.data[c];
        }
        let last = (n - 1) as f64;
        if frame >= last {
            return self.data[(n - 1) * self.channels as usize + c];
        }
        let i = frame.floor();
        let frac = (frame - i) as f32;
        let i0 = i as usize;
        let ch = self.channels as usize;
        let a = self.data[i0 * ch + c];
        let b = self.data[(i0 + 1) * ch + c];
        a + (b - a) * frac
    }
}

pub struct AudioReader;

impl AudioReader {
    pub fn load(path: &Path) -> Result<AudioBuffer> {
        let mut reader = WavReader::open(path)?;
        let spec = reader.spec();
        let channels = spec.channels;
        let sample_rate = spec.sample_rate;
        let mut data = Vec::new();
        match spec.sample_format {
            SampleFormat::Float => {
                for s in reader.samples::<f32>() {
                    data.push(s?);
                }
            }
            SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                for s in reader.samples::<i32>() {
                    data.push(s? as f32 / max);
                }
            }
        }
        Ok(AudioBuffer {
            sample_rate,
            channels,
            data,
        })
    }
}

/// Write 32-bit float WAV (interleaved). Number of frames derived from data length.
pub fn write_wav_f32(path: &Path, buf: &AudioBuffer) -> Result<()> {
    let spec = WavSpec {
        channels: buf.channels,
        sample_rate: buf.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for s in &buf.data {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Assert every track claims the same nominal sample rate. Actual clock
/// differences are modelled separately by the time map.
pub fn check_nominal_rates(rates: &[u32]) -> Result<()> {
    if let Some(first) = rates.first() {
        for r in rates {
            if r != first {
                return Err(AlignError::IncompatibleRate(*first, *r));
            }
        }
    }
    Ok(())
}
