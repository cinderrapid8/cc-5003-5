use crate::envelope::{adaptive_threshold, envelope};

/// A detected onset (transient/clap) in seconds from the start of the track.
#[derive(Clone, Debug)]
pub struct Onset {
    pub time: f64,
    pub strength: f32,
}

/// One-pole high-pass pre-emphasis. Sustained tonal material (e.g. a long
/// rehearsal note) is suppressed while broadband claps and drum attacks pass.
pub fn high_pass_pub(x: &[f32], sample_rate: u32, cutoff_hz: f32) -> Vec<f32> {
    high_pass(x, sample_rate, cutoff_hz)
}

fn high_pass(x: &[f32], sample_rate: u32, cutoff_hz: f32) -> Vec<f32> {
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let dt = 1.0 / sample_rate as f32;
    let alpha = rc / (rc + dt);
    let mut out = vec![0f32; x.len()];
    let mut prev_x = 0f32;
    let mut prev_y = 0f32;
    for (i, s) in x.iter().enumerate() {
        prev_y = alpha * (prev_y + *s - prev_x);
        prev_x = *s;
        out[i] = prev_y;
    }
    out
}

/// Detect transients suitable as clap/beat reference anchors.
///
/// The analysis signal is high-pass filtered before envelope extraction so a
/// sustained tone cannot mask short transients. Positive spectral flux over an
/// adaptive local threshold is combined with a prominence requirement and
/// non-maximum suppression. Returns times in seconds.
pub fn detect_onsets(mono: &[f32], sample_rate: u32, min_spacing_secs: f64) -> Vec<Onset> {
    let filtered = high_pass(mono, sample_rate, 300.0);
    let env = envelope(&filtered, sample_rate);
    let thresh = adaptive_threshold(&env, 0.4, sample_rate, 1.2);

    let min_gap = (min_spacing_secs * sample_rate as f64) as usize;
    let mut flux = vec![0f32; env.len()];
    for i in 1..env.len() {
        let d = env[i] - env[i - 1];
        if d > 0.0 {
            flux[i] = d;
        }
    }

    let noise = noise_floor(&flux);
    let mut peaks: Vec<Onset> = Vec::new();
    let mut i = 1;
    while i + 1 < flux.len() {
        // Prominence: the envelope must jump clearly above its local floor,
        // independent of the absolute level of any sustained tone.
        let prominent = env[i] > thresh[i] && env[i] > 0.02;
        if flux[i] > flux[i - 1]
            && flux[i] >= flux[i + 1]
            && prominent
            && flux[i] > noise * 2.5
        {
            // Refine to the local maximum within +-5 ms.
            let refine = (0.005 * sample_rate as f64) as usize;
            let start = i.saturating_sub(refine);
            let end = (i + refine + 1).min(flux.len());
            let mut best = i;
            for j in start..end {
                if flux[j] > flux[best] {
                    best = j;
                }
            }
            peaks.push(Onset {
                time: best as f64 / sample_rate as f64,
                strength: flux[best],
            });
            i = best + min_gap.max(1);
        } else {
            i += 1;
        }
    }
    peaks
}

fn noise_floor(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2].max(1e-9)
}
