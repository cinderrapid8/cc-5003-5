use crate::audio::AudioBuffer;
use crate::map::TimeMap;
use serde::{Deserialize, Serialize};

/// Ground truth used by synthetic tests.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroundTruth {
    pub delay_secs: f64,
    pub rate_ratio: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EstimatorError {
    pub delay_error_secs: f64,
    pub drift_error_ppm: f64,
    pub rms_residual_secs: f64,
    pub max_abs_residual_secs: f64,
}

/// Compare a fitted time map against the known generative clock model over a
/// grid of times spanning the full recording, including segment joins.
///
/// Convention shared with `synth::ClockModel`: a physical event at reference
/// time `t` is stamped on the member's own clock at `(t + delay) * ratio`.
/// Solving for the project (reference) time gives
/// `project = member / ratio - delay`, so the true map slope is `1/ratio`
/// and the true intercept is `-delay`.
pub fn estimator_error(map: &TimeMap, truth: &GroundTruth, duration: f64) -> EstimatorError {
    let true_slope = 1.0 / truth.rate_ratio;
    let true_offset = -truth.delay_secs;

    let n = 200;
    let mut sq = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut slope_err_max = 0.0f64;
    let mut delay_at_zero_err = 0.0f64;
    for i in 0..=n {
        let member_time = duration * i as f64 / n as f64;
        let true_project = true_slope * member_time + true_offset;
        let got = map.to_project(member_time).unwrap_or(f64::NAN);
        let e = got - true_project;
        sq += e * e;
        max_abs = max_abs.max(e.abs());
    }
    for s in &map.segments {
        slope_err_max = slope_err_max.max((s.slope - true_slope).abs());
        // Delay estimate read at the segment start: project = slope*member +
        // offset, and at member=0 an ideal slope-1 map intercepts at -delay.
        let true_p0 = true_slope * s.t0 + true_offset;
        delay_at_zero_err = delay_at_zero_err.max((s.project_t0 - true_p0).abs());
    }
    EstimatorError {
        delay_error_secs: delay_at_zero_err,
        drift_error_ppm: slope_err_max * 1e6,
        rms_residual_secs: (sq / (n as f64 + 1.0)).sqrt(),
        max_abs_residual_secs: max_abs,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhaseContinuityReport {
    /// Maximum phase jump (in radians) detected at segment joins, measured on
    /// the rendered output's instantaneous phase.
    pub max_phase_jump_rad: f64,
    /// Position of the worst join in seconds.
    pub worst_join_secs: f64,
}

/// Track the phase of a sinusoidal component through a rendered buffer and
/// report the largest discontinuity. A continuous time map must not introduce
/// a phase jump beyond what the windowed-sinc kernel itself produces.
pub fn phase_continuity(
    rendered: &AudioBuffer,
    freq_hz: f64,
    join_times: &[f64],
) -> PhaseContinuityReport {
    let mono = rendered.mono_mix();
    let rate = rendered.sample_rate as f64;
    let mut analytic_phase = Vec::with_capacity(mono.len());
    let mut prev_phase = 0.0f64;
    let step = 2.0 * std::f64::consts::PI * freq_hz / rate;
    // Demodulate at the expected frequency and unwrap.
    let mut unwrapped = vec![0f64; mono.len()];
    let mut cum = 0.0;
    for (i, s) in mono.iter().enumerate() {
        let t = i as f64 / rate;
        let lo = (2.0 * std::f64::consts::PI * freq_hz * t).cos();
        let quadr = (-2.0 * std::f64::consts::PI * freq_hz * t).sin();
        let v = *s as f64;
        let re = v * lo;
        let im = v * quadr;
        let ph = im.atan2(re);
        if i > 0 {
            let mut d = ph - prev_phase;
            while d > std::f64::consts::PI {
                d -= 2.0 * std::f64::consts::PI;
            }
            while d < -std::f64::consts::PI {
                d += 2.0 * std::f64::consts::PI;
            }
            cum += d;
        }
        prev_phase = ph;
        unwrapped[i] = cum;
        analytic_phase.push(ph);
    }

    let mut max_jump = 0.0f64;
    let mut worst = 0.0f64;
    let window = (0.01 * rate) as usize; // 10 ms each side
    for j in join_times {
        let idx = (j * rate) as isize;
        let lo = (idx - window as isize).max(1) as usize;
        let hi = (idx + window as isize).min(mono.len() as isize - 1) as usize;
        if hi <= lo + 2 {
            continue;
        }
        // Compare local unwrapped-phase slopes; a jump shows as an offset of
        // the phase residuals relative to a straight line.
        let before = &unwrapped[lo..idx as usize];
        let after = &unwrapped[idx as usize..hi];
        if before.len() < 4 || after.len() < 4 {
            continue;
        }
        let mb = mean(before);
        let ma = mean(after);
        let jump = (ma - mb).abs();
        if jump > max_jump {
            max_jump = jump;
            worst = *j;
        }
    }
    let _ = step;
    PhaseContinuityReport {
        max_phase_jump_rad: max_jump,
        worst_join_secs: worst,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelConsistencyReport {
    /// Max absolute sample difference between channels after alignment, where
    /// the source material was identical across channels. Must be at kernel
    /// noise level, proving one shared time map drives every channel.
    pub max_interchannel_diff: f64,
    pub rms_interchannel_diff: f64,
}

/// When a source was rendered with identical content in every channel, the
/// aligned output must preserve that identity exactly (same map, same kernel).
pub fn channel_consistency(rendered: &AudioBuffer) -> ChannelConsistencyReport {
    let ch = rendered.channels as usize;
    let frames = rendered.frames();
    if ch < 2 {
        return ChannelConsistencyReport {
            max_interchannel_diff: 0.0,
            rms_interchannel_diff: 0.0,
        };
    }
    let mut max_diff = 0.0f64;
    let mut sq = 0.0;
    let mut count = 0u64;
    for f in 0..frames {
        let first = rendered.data[f * ch] as f64;
        for c in 1..ch {
            let d = (rendered.data[f * ch + c] as f64 - first).abs();
            max_diff = max_diff.max(d);
            sq += d * d;
            count += 1;
        }
    }
    ChannelConsistencyReport {
        max_interchannel_diff: max_diff,
        rms_interchannel_diff: (sq / count.max(1) as f64).sqrt(),
    }
}

fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}
