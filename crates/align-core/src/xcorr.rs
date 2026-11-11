use crate::envelope::{adaptive_threshold, envelope};
use rustfft::{num_complex::Complex, FftPlanner};

/// One cross-correlation peak candidate.
#[derive(Clone, Debug)]
pub struct CorrPeak {
    /// Delay in samples: with `x` the reference window and `y` the member
    /// window, a peak at delay d means `y[t + d] ~= x[t]`, i.e. the member
    /// event sits d samples later than the reference event. Positive values
    /// therefore mean the member lags the reference, matching
    /// `member_time = ref_time + delay`.
    pub lag_samples: f64,
    /// Normalized correlation in [-1, 1] at the peak.
    pub score: f64,
}

/// Full normalized cross-correlation of `x` and `y` over integer lags
/// `[-max_lag, max_lag]`, computed with one FFT convolution.
///
/// Returns scores indexed by lag + max_lag. Correlation at each lag is
/// energy-normalized over the actual overlap, so differing gains do not
/// inflate the score.
pub fn normalized_xcorr(x: &[f32], y: &[f32], max_lag: usize) -> Vec<f64> {
    let n = x.len() + y.len() - 1;
    let fft_len = n.next_power_of_two();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let mut fx = vec![Complex::new(0f32, 0.0); fft_len];
    let mut fy = vec![Complex::new(0f32, 0.0); fft_len];
    for (i, s) in x.iter().enumerate() {
        fx[i] = Complex::new(*s, 0.0);
    }
    // Correlation = conv(x, reversed y).
    for (i, s) in y.iter().enumerate() {
        fy[y.len() - 1 - i] = Complex::new(*s, 0.0);
    }
    fft.process(&mut fx);
    fft.process(&mut fy);
    let mut fprod: Vec<Complex<f32>> = fx
        .iter()
        .zip(fy.iter())
        .map(|(a, b)| a * b)
        .collect();
    ifft.process(&mut fprod);
    let scale = 1.0 / fft_len as f32;

    // conv[k] corresponds to lag k - (y.len() - 1).
    let offset = y.len() as isize - 1;
    let lags = (max_lag as isize * 2 + 1) as usize;
    let mut out = vec![0f64; lags];

    // Precompute prefix sums of squared energy for overlap normalization.
    let sx2 = prefix_sq(x);
    let sy2 = prefix_sq(y);
    let mean_x = x.iter().sum::<f32>() / x.len().max(1) as f32;
    let mean_y = y.iter().sum::<f32>() / y.len().max(1) as f32;

    for (idx, slot) in out.iter_mut().enumerate() {
        let lag = idx as isize - max_lag as isize;
        let k = (lag + offset) as usize;
        let raw = fprod[k].re * scale;

        // Overlap ranges in x and y for this lag.
        let (x0, x1, y0, y1) = overlap_ranges(x.len(), y.len(), lag);
        let len = (x1 - x0).max(1) as f32;
        let ex = energy(&sx2, x0, x1) - mean_x * mean_x * len;
        let ey = energy(&sy2, y0, y1) - mean_y * mean_y * len;
        let denom = (ex.max(0.0) * ey.max(0.0)).sqrt();
        // Remove the mean component from the raw dot product as well.
        let mean_corr = mean_x * mean_y * len;
        let centered = (raw - mean_corr) as f64;
        *slot = if denom > 1e-12 {
            centered / denom as f64
        } else {
            0.0
        };
    }
    out
}

/// Given lag where x[t] ~= y[t + lag], return overlapping index ranges.
fn overlap_ranges(nx: usize, ny: usize, lag: isize) -> (usize, usize, usize, usize) {
    // t in [0, nx), t+lag in [0, ny)
    let t_lo = (-lag).max(0) as usize;
    let t_hi = ((ny as isize - lag).min(nx as isize).max(0)) as usize;
    let y_lo = (t_lo as isize + lag).max(0) as usize;
    let y_hi = (t_hi as isize + lag) as usize;
    (t_lo, t_hi, y_lo, y_hi)
}

fn prefix_sq(x: &[f32]) -> Vec<f64> {
    let mut p = vec![0f64; x.len() + 1];
    for (i, s) in x.iter().enumerate() {
        p[i + 1] = p[i] + (*s as f64) * (*s as f64);
    }
    p
}

fn energy(p: &[f64], lo: usize, hi: usize) -> f32 {
    (p[hi] - p[lo]) as f32
}

/// Find the `k` best distinct correlation peaks with parabolic sub-sample
/// refinement. Peaks closer than `min_distance` lags are merged, strongest wins.
pub fn top_peaks(scores: &[f64], max_lag: usize, k: usize, min_distance: usize) -> Vec<CorrPeak> {
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    for i in 1..scores.len().saturating_sub(1) {
        if scores[i] > scores[i - 1] && scores[i] >= scores[i + 1] && scores[i] > 0.0 {
            candidates.push((i, scores[i]));
        }
    }
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut picked: Vec<(usize, f64)> = Vec::new();
    for c in candidates {
        if picked
            .iter()
            .all(|p| (p.0 as isize - c.0 as isize).abs() > min_distance as isize)
        {
            picked.push(c);
        }
        if picked.len() >= k {
            break;
        }
    }

    picked
        .into_iter()
        .map(|(i, s)| {
            let (delta, refined_score) = parabolic_interp(scores, i);
            // Correlation index k corresponds to conv lag `k - max_lag`, where
            // a positive conv lag means y is shifted earlier. The member delay
            // convention (member later than reference) is the negation.
            let conv_lag = (i as isize - max_lag as isize) as f64 + delta;
            CorrPeak {
                lag_samples: -conv_lag,
                score: refined_score.max(s),
            }
        })
        .collect()
}

/// Parabolic interpolation around index `i`. Returns (fractional offset, peak value).
fn parabolic_interp(v: &[f64], i: usize) -> (f64, f64) {
    if i == 0 || i + 1 >= v.len() {
        return (0.0, v[i]);
    }
    let a = v[i - 1];
    let b = v[i];
    let c = v[i + 1];
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-12 {
        return (0.0, b);
    }
    let delta = 0.5 * (a - c) / denom;
    let peak = b - 0.25 * (a - c) * delta;
    (delta, peak)
}

/// Transient-only analysis signal: positive flux of the high-pass envelope.
/// Sustained tones cancel out, so correlation locks onto claps and attacks
/// rather than the loudest continuous material.
pub fn transient_flux(x: &[f32], sample_rate: u32) -> Vec<f32> {
    let filtered = crate::onset::high_pass_pub(x, sample_rate, 300.0);
    let env = envelope(&filtered, sample_rate);
    let _ = adaptive_threshold(&env, 0.4, sample_rate, 1.2);
    let mut flux = vec![0f32; env.len()];
    for i in 1..env.len() {
        flux[i] = (env[i] - env[i - 1]).max(0.0);
    }
    flux
}

/// Coarse global delay between two full tracks, in seconds. Correlates the
/// transient-flux signals over the complete recording so the result is not
/// biased toward any single segment. Returns the best and alternative peaks.
pub fn estimate_coarse_delay(
    reference: &[f32],
    member: &[f32],
    sample_rate: u32,
    max_delay_secs: f64,
) -> Vec<CorrPeak> {
    let rf = transient_flux(reference, sample_rate);
    let mf = transient_flux(member, sample_rate);
    let max_lag = (max_delay_secs * sample_rate as f64) as usize;
    let scores = normalized_xcorr(&rf, &mf, max_lag);
    top_peaks(
        &scores,
        max_lag,
        5,
        (0.05 * sample_rate as f64) as usize,
    )
}

/// Result of a local correlation search.
#[derive(Clone, Debug)]
pub struct LocalMatch {
    /// Fractional position on the member timeline of the matched event.
    pub member_time: f64,
    /// Normalized correlation score in [-1, 1].
    pub score: f64,
    /// Alternative candidate positions (for user selection).
    pub candidates: Vec<CorrPeak>,
}

/// Correlate a window of the reference around `ref_center` against a window
/// of the member around `expected_member_time`. The search zero-lag is placed
/// at the expected member position, so the lag range covers realistic fixed
/// delays instead of being pinned to the reference timestamp.
///
/// Returned `CorrPeak.lag_samples` is expressed relative to the expected
/// member position; `LocalMatch.member_time` converts it to an absolute
/// member-timeline timestamp.
pub fn local_correlation(
    reference: &[f32],
    member: &[f32],
    sample_rate: u32,
    ref_center: f64,
    expected_member_time: f64,
    window_secs: f64,
    max_lag_secs: f64,
) -> Option<LocalMatch> {
    let w = (window_secs * sample_rate as f64 / 2.0) as usize;
    let max_lag = (max_lag_secs * sample_rate as f64) as usize;
    let rc = (ref_center * sample_rate as f64) as isize;
    let mc = (expected_member_time * sample_rate as f64) as isize;
    let r0 = (rc - w as isize).max(0) as usize;
    let r1 = (rc + w as isize).min(reference.len() as isize) as usize;
    let m0 = (mc - w as isize).max(0) as usize;
    let m1 = (mc + w as isize).min(member.len() as isize) as usize;
    if r1 - r0 < 8 || m1 - m0 < 8 {
        return None;
    }
    let r_flux = transient_flux(&reference[r0..r1], sample_rate);
    let m_flux = transient_flux(&member[m0..m1], sample_rate);
    let lag_cap = max_lag.min(w / 2);
    let scores = normalized_xcorr(&r_flux, &m_flux, lag_cap);
    let peaks = top_peaks(
        &scores,
        lag_cap,
        5,
        (0.02 * sample_rate as f64) as usize,
    );
    let best = peaks.first()?;
    let member_time = expected_member_time + best.lag_samples / sample_rate as f64;
    Some(LocalMatch {
        member_time,
        score: best.score,
        candidates: peaks,
    })
}
