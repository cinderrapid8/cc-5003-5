use serde::{Deserialize, Serialize};

use crate::audio::AudioBuffer;
use crate::error::Result;
use crate::map::TimeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderOptions {
    /// Output sample rate of the aligned project (typically the nominal rate).
    pub output_rate: u32,
    /// Half-length of the windowed-sinc kernel in output frames.
    pub kernel_taps_half: usize,
    /// If true, regions flagged unknown in the time map are still rendered by
    /// extrapolation but marked in the report. If false, they are zero-filled.
    pub render_unknown_preview: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            output_rate: 48_000,
            kernel_taps_half: 32,
            render_unknown_preview: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderReport {
    pub track_id: String,
    pub output_frames: usize,
    pub output_duration_secs: f64,
    /// Output frames before the first mapped source sample (zero filled).
    pub head_pad_frames: usize,
    /// Output frames after the last mapped source sample (zero filled).
    pub tail_pad_frames: usize,
    /// Frames rendered from time-map regions marked unknown.
    pub unknown_frames: usize,
    /// Frames that needed edge clamping because the inverse map landed on or
    /// beyond the source boundary.
    pub edge_clamped_frames: usize,
    pub min_map_slope: f64,
    pub max_map_slope: f64,
}

/// Rendered aligned track plus its audit report.
pub struct RenderedTrack {
    pub buffer: AudioBuffer,
    pub report: RenderReport,
}

/// Resample one track onto the shared project timeline.
///
/// Length rule: the output covers `[0, project_end]` where `project_end` is
/// the largest mapped end over all tracks supplied to `render_project`. For a
/// single track call, its own mapped end is used. Output frames whose inverse
/// map falls outside the source are zero filled; the counts are reported.
pub fn render_track(
    source: &AudioBuffer,
    map: &TimeMap,
    project_end_secs: Option<f64>,
    opts: &RenderOptions,
) -> Result<RenderedTrack> {
    let (_, own_end) = map
        .project_span()
        .ok_or_else(|| crate::error::AlignError::Other("time map has no segments".into()))?;
    let project_end = project_end_secs.unwrap_or(own_end).max(0.0);
    let out_frames = (project_end * opts.output_rate as f64).ceil() as usize;

    let mut out = AudioBuffer::new(opts.output_rate, source.channels);
    out.data = vec![0f32; out_frames * source.channels as usize];

    let mut report = RenderReport {
        track_id: map.track_id.clone(),
        output_frames: out_frames,
        output_duration_secs: out_frames as f64 / opts.output_rate as f64,
        head_pad_frames: 0,
        tail_pad_frames: 0,
        unknown_frames: 0,
        edge_clamped_frames: 0,
        min_map_slope: f64::INFINITY,
        max_map_slope: f64::NEG_INFINITY,
    };
    for s in &map.segments {
        report.min_map_slope = report.min_map_slope.min(s.slope);
        report.max_map_slope = report.max_map_slope.max(s.slope);
    }

    let source_frames = source.frames();
    let src_rate = source.sample_rate as f64;
    let out_rate = opts.output_rate as f64;
    let half = opts.kernel_taps_half;

    for frame in 0..out_frames {
        let project_time = frame as f64 / out_rate;
        let member_time = map.to_member(project_time)?;
        let known = map.is_known_at(member_time.clamp(0.0, source.duration_secs()));
        if !known {
            report.unknown_frames += 1;
            if !opts.render_unknown_preview {
                continue;
            }
        }
        if member_time < 0.0 {
            report.head_pad_frames += 1;
            continue;
        }
        if member_time >= source.duration_secs() {
            report.tail_pad_frames += 1;
            continue;
        }
        let src_pos = member_time * src_rate;
        let center = src_pos.floor() as isize;
        let frac = (src_pos - center as f64) as f32;

        for ch in 0..source.channels as usize {
            let mut acc = 0f32;
            let mut norm = 0f32;
            let mut clamped = false;
            for k in -(half as isize)..=half as isize {
                let idx = center + k;
                let sample = if idx < 0 || idx as usize >= source_frames {
                    clamped = true;
                    if idx < 0 {
                        source.data[ch]
                    } else {
                        source.data[(source_frames - 1) * source.channels as usize + ch]
                    }
                } else {
                    source.data[idx as usize * source.channels as usize + ch]
                };
                let kernel = sinc_kernel(k as f64, frac as f64, half as f64);
                acc += sample * kernel;
                norm += kernel;
            }
            if norm.abs() > 1e-6 {
                acc /= norm;
            }
            if clamped {
                report.edge_clamped_frames += 1;
            }
            out.data[frame * source.channels as usize + ch] = acc;
        }
    }
    // Edge clamping was counted per channel; normalize to per-frame.
    report.edge_clamped_frames /= source.channels.max(1) as usize;
    Ok(RenderedTrack { buffer: out, report })
}

/// Windowed-sinc interpolation kernel. `k` is the integer distance from the
/// fractional position, `frac` in [0, 1), `half` is the kernel half-length.
/// The Hann window reaches zero at both kernel ends, which keeps the
/// interpolation band-limited without edge discontinuity ringing.
fn sinc_kernel(k: f64, frac: f64, half: f64) -> f32 {
    let x = k - frac;
    if x.abs() < 1e-12 {
        return 1.0;
    }
    if x.abs() >= half {
        return 0.0;
    }
    let px = std::f64::consts::PI * x;
    let sinc = px.sin() / px;
    // Hann over [-half, half], zero at both ends.
    let window = 0.5 * (1.0 + (std::f64::consts::PI * x / half).cos());
    (sinc * window) as f32
}

/// Render every track against one shared project end so the outputs share a
/// common length and can be mixed or edited as separate stems.
pub fn render_project(
    sources: &[(&AudioBuffer, &TimeMap)],
    opts: &RenderOptions,
) -> Result<(Vec<RenderedTrack>, f64)> {
    let mut project_end = 0.0f64;
    for (_, map) in sources {
        if let Some((_, end)) = map.project_span() {
            project_end = project_end.max(end);
        }
    }
    project_end = project_end.max(0.0);
    let mut tracks = Vec::with_capacity(sources.len());
    for (buf, map) in sources {
        tracks.push(render_track(buf, map, Some(project_end), opts)?);
    }
    Ok((tracks, project_end))
}

/// Simple equal-gain mix for an A/B preview file.
pub fn mix_stems(tracks: &[&AudioBuffer]) -> AudioBuffer {
    let rate = tracks.first().map(|t| t.sample_rate).unwrap_or(48_000);
    let channels = tracks.first().map(|t| t.channels).unwrap_or(1);
    let frames = tracks.iter().map(|t| t.frames()).max().unwrap_or(0);
    let mut out = AudioBuffer::new(rate, channels);
    out.data = vec![0f32; frames * channels as usize];
    for t in tracks {
        for (i, s) in t.data.iter().enumerate() {
            out.data[i] += *s;
        }
    }
    let gain = 1.0 / tracks.len().max(1) as f32;
    for s in out.data.iter_mut() {
        *s *= gain;
    }
    out
}
