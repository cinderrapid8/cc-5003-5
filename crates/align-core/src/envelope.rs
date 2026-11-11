/// Rectified, low-pass smoothed envelope for onset detection.
/// Operates on mono analysis signals at one nominal sample rate.
pub fn envelope(x: &[f32], sample_rate: u32) -> Vec<f32> {
    let n = x.len();
    let mut env = vec![0f32; n];
    // ~5 ms one-pole smoothing for attack, ~50 ms for release.
    let attack = (0.005 * sample_rate as f32).max(1.0);
    let release = (0.05 * sample_rate as f32).max(1.0);
    let a_up = 1.0 - (-1.0 / attack).exp();
    let a_dn = 1.0 - (-1.0 / release).exp();
    let mut e = 0f32;
    for i in 0..n {
        let r = x[i].abs();
        let coef = if r > e { a_up } else { a_dn };
        e += coef * (r - e);
        env[i] = e;
    }
    env
}

/// Adaptive local threshold: sliding-window mean of the envelope, scaled by k.
pub fn adaptive_threshold(env: &[f32], window_secs: f64, sample_rate: u32, k: f32) -> Vec<f32> {
    let w = (window_secs * sample_rate as f64 / 2.0).max(1.0) as usize;
    let n = env.len();
    let mut out = vec![0f32; n];
    let mut sum = 0f64;
    let mut lo = 0usize;
    let mut hi = 0usize; // exclusive
    for (i, slot) in out.iter_mut().enumerate() {
        let target_lo = i.saturating_sub(w);
        let target_hi = (i + w + 1).min(n);
        while hi < target_hi {
            sum += env[hi] as f64;
            hi += 1;
        }
        while lo < target_lo {
            sum -= env[lo] as f64;
            lo += 1;
        }
        let len = (hi - lo) as f64;
        *slot = (sum / len) as f32 * k;
    }
    out
}
