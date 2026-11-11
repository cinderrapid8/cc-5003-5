use crate::audio::AudioBuffer;

/// Parameters describing how a member recorder deviates from the reference
/// clock. All ground truth is expressed through these values so tests can
/// check estimator error directly.
#[derive(Clone, Debug)]
pub struct ClockModel {
    /// Fixed delay in seconds: member event time = ref event time + delay
    /// before drift is applied.
    pub delay_secs: f64,
    /// Clock ratio member/ref. 1.0 + 50e-6 means the member clock runs 50 ppm
    /// fast, so more samples are produced for the same physical interval.
    pub rate_ratio: f64,
    /// Nominal sample rate both files claim (actual rates differ by ratio).
    pub nominal_rate: u32,
}

impl ClockModel {
    /// Map a physical/reference time to the member recorder's own timeline.
    pub fn member_time(&self, ref_time: f64) -> f64 {
        (ref_time + self.delay_secs) * self.rate_ratio
    }
}

/// One scheduled hit (clap or drum) on the reference timeline.
#[derive(Clone, Debug)]
pub struct Hit {
    pub time: f64,
    pub kind: HitKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HitKind {
    /// Short broadband burst — models a hand clap shared by all recordings.
    Clap,
    /// Pitched drum hit with a distinct spectral center.
    Drum(f32),
}

/// A gap in the reference: events inside these intervals are not recorded on
/// the reference track, so anchors there must end up unknown rather than
/// force-fit.
#[derive(Clone, Debug)]
pub struct Gap {
    pub start: f64,
    pub end: f64,
}

impl Gap {
    pub fn contains(&self, t: f64) -> bool {
        t >= self.start && t <= self.end
    }
}

/// Render a reference track: hits plus an optional sustained tone. Gaps are
/// zeroed (clips are not removed — silence models a dropped segment).
pub fn render_reference(
    hits: &[Hit],
    tone: Option<(f32, f64, f64)>, // (freq, start, end)
    gaps: &[Gap],
    duration: f64,
    rate: u32,
    channels: u16,
) -> AudioBuffer {
    let frames = (duration * rate as f64) as usize;
    let mut buf = AudioBuffer::new(rate, channels);
    buf.data = vec![0f32; frames * channels as usize];
    if let Some((freq, start, end)) = tone {
        add_tone(&mut buf, freq, start, end, 0.2);
    }
    for h in hits {
        if gaps.iter().any(|g| g.contains(h.time)) {
            continue;
        }
        match h.kind {
            HitKind::Clap => add_clap(&mut buf, h.time, 0.5),
            HitKind::Drum(f) => add_drum(&mut buf, h.time, f, 0.5),
        }
    }
    apply_gaps(&mut buf, gaps);
    buf
}

/// Render a member track under a known clock model. The member produces
/// samples at its own (deviating) rate but tags them with the nominal rate;
/// events are placed at `model.member_time(...)`. An interruption zeroes a
/// member-local interval, simulating a recorder stop/restart.
pub fn render_member(
    hits: &[Hit],
    tone: Option<(f32, f64, f64)>,
    model: &ClockModel,
    interruptions: &[Gap],
    ref_duration: f64,
    channels: u16,
) -> AudioBuffer {
    let member_duration = model.member_time(ref_duration);
    let frames = (member_duration * model.nominal_rate as f64) as usize;
    let mut buf = AudioBuffer::new(model.nominal_rate, channels);
    buf.data = vec![0f32; frames * channels as usize];
    if let Some((freq, start, end)) = tone {
        let ms = model.member_time(start);
        let me = model.member_time(end);
        // Tone frequency is generated at the member's real rate, so a fixed
        // physical pitch appears shifted by rate_ratio on its nominal clock.
        add_tone(&mut buf, freq / model.rate_ratio as f32, ms, me, 0.2);
    }
    for h in hits {
        let t = model.member_time(h.time);
        if interruptions.iter().any(|g| g.contains(t)) {
            continue;
        }
        match h.kind {
            HitKind::Clap => add_clap(&mut buf, t, 0.5),
            HitKind::Drum(f) => add_drum(&mut buf, t, f / model.rate_ratio as f32, 0.5),
        }
    }
    apply_gaps(&mut buf, interruptions);
    buf
}

fn add_tone(buf: &mut AudioBuffer, freq: f32, start: f64, end: f64, amp: f32) {
    let rate = buf.sample_rate as f64;
    let f0 = (start * rate) as usize;
    let f1 = ((end * rate) as usize).min(buf.frames());
    let ch = buf.channels as usize;
    let mut phase = 0f32;
    let step = 2.0 * std::f32::consts::PI * freq / buf.sample_rate as f32;
    for f in f0..f1 {
        phase += step;
        let s = amp * phase.sin();
        for c in 0..ch {
            buf.data[f * ch + c] += s;
        }
    }
}

fn add_clap(buf: &mut AudioBuffer, at: f64, amp: f32) {
    let rate = buf.sample_rate as f64;
    let start = (at * rate) as isize;
    let len = (0.03 * rate) as usize; // 30 ms noise burst
    let ch = buf.channels as usize;
    // Deterministic pseudo-noise so tests are reproducible.
    let mut seed = (start.max(1) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for k in 0..len {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let noise = ((seed >> 40) as f32 / (1u32 << 24) as f32) - 0.5;
        let env = (1.0 - k as f32 / len as f32).powi(2);
        let idx = start + k as isize;
        if idx < 0 || idx as usize >= buf.frames() {
            continue;
        }
        for c in 0..ch {
            buf.data[idx as usize * ch + c] += amp * 2.0 * noise * env;
        }
    }
}

fn add_drum(buf: &mut AudioBuffer, at: f64, freq: f32, amp: f32) {
    let rate = buf.sample_rate as f64;
    let start = (at * rate) as isize;
    let len = (0.12 * rate) as usize;
    let ch = buf.channels as usize;
    let mut phase = 0f32;
    for k in 0..len {
        let idx = start + k as isize;
        if idx < 0 || idx as usize >= buf.frames() {
            continue;
        }
        // Pitch sweep downward, like a kick/snare body.
        let f = freq * (1.6 - 0.8 * k as f32 / len as f32);
        phase += 2.0 * std::f32::consts::PI * f / buf.sample_rate as f32;
        let env = (1.0 - k as f32 / len as f32).powi(2);
        let s = amp * phase.sin() * env;
        for c in 0..ch {
            buf.data[idx as usize * ch + c] += s;
        }
    }
}

fn apply_gaps(buf: &mut AudioBuffer, gaps: &[Gap]) {
    let rate = buf.sample_rate as f64;
    let ch = buf.channels as usize;
    for g in gaps {
        let f0 = (g.start * rate) as usize;
        let f1 = ((g.end * rate) as usize).min(buf.frames());
        for f in f0..f1 {
            for c in 0..ch {
                buf.data[f * ch + c] = 0.0;
            }
        }
    }
}
