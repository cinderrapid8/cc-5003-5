use crate::anchor::{Anchor, AnchorStatus};
use crate::audio::AudioBuffer;
use crate::error::Result;
use crate::onset::detect_onsets;
use crate::synth::Gap;
use crate::xcorr::{estimate_coarse_delay, local_correlation};

/// Tunables for anchor detection.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Minimum spacing between reference events treated as potential anchors.
    pub min_onset_spacing_secs: f64,
    /// Correlation window length around each event.
    pub window_secs: f64,
    /// How far from the expected position a local search may reach.
    pub max_lag_secs: f64,
    /// Score below which an anchor is held as Unknown instead of trusted.
    pub min_confidence: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            min_onset_spacing_secs: 0.25,
            window_secs: 0.2,
            max_lag_secs: 1.0,
            min_confidence: 0.35,
        }
    }
}

/// Detect anchors for one member track against the reference.
///
/// Anchors are distributed across the whole recording: every reference onset
/// gets its own independent local correlation. Regions where the reference is
/// silent (a gap) or the member is interrupted produce Unknown anchors rather
/// than forced matches, so low-confidence parts never overwrite the real
/// performance timing.
pub fn detect_anchors(
    track_id: &str,
    reference: &AudioBuffer,
    member: &AudioBuffer,
    ref_gaps: &[Gap],
    member_gaps: &[Gap],
    config: &PipelineConfig,
) -> Result<Vec<Anchor>> {
    let ref_mono = reference.mono_mix();
    let mem_mono = member.mono_mix();

    let ref_onsets = detect_onsets(
        &ref_mono,
        reference.sample_rate,
        config.min_onset_spacing_secs,
    );

    // Global coarse delay from transient flux across the whole recording.
    // This anchors the local searches near the true offset instead of the
    // reference timestamp, and yields the candidate list users can pick from
    // when repeated beats produce multiple correlation peaks.
    let coarse = estimate_coarse_delay(
        &ref_mono,
        &mem_mono,
        reference.sample_rate,
        config.max_lag_secs,
    );
    let coarse_delay = coarse
        .first()
        .map(|p| p.lag_samples / reference.sample_rate as f64)
        .unwrap_or(0.0);

    let mut anchors = Vec::new();
    let mut n = 0usize;
    for onset in &ref_onsets {
        n += 1;
        let id = format!("{track_id}-a{n:03}");

        // Margin covers onsets inside the correlation window that overlap the
        // edge of a gap, not just those exactly inside it.
        let in_ref_gap = ref_gaps.iter().any(|g| {
            onset.time >= g.start - config.window_secs && onset.time <= g.end + config.window_secs
        });
        if in_ref_gap {
            anchors.push(unknown_anchor(id.clone(), track_id, onset.time));
            continue;
        }

        // Expected member position from the global delay estimate. Drift makes
        // the true lag drift slowly over time; the local window absorbs that.
        let expected = (onset.time + coarse_delay).clamp(0.0, member.duration_secs());
        let Some(m) = local_correlation(
            &ref_mono,
            &mem_mono,
            reference.sample_rate,
            onset.time,
            expected,
            config.window_secs,
            // Local search only needs to cover drift accumulation + jitter.
            (config.window_secs * 0.5).min(0.25),
        ) else {
            anchors.push(unknown_anchor(id, track_id, onset.time));
            continue;
        };

        if m.score < config.min_confidence {
            let mut a = anchor_from_match(
                id,
                track_id,
                onset.time,
                &m,
                reference.sample_rate,
                expected,
            );
            a.mark_unknown();
            anchors.push(a);
            continue;
        }

        let a = anchor_from_match(
            id,
            track_id,
            onset.time,
            &m,
            reference.sample_rate,
            expected,
        );

        // If the chosen match lands near or inside a member interruption,
        // keep it unknown even if the raw score is decent.
        let mut a = a;
        let near_member_gap = member_gaps.iter().any(|g| {
            a.member_time >= g.start - config.window_secs
                && a.member_time <= g.end + config.window_secs
        });
        if near_member_gap {
            a.mark_unknown();
            a.note = Some("member interruption: event not recorded".into());
        }
        anchors.push(a);
    }
    Ok(anchors)
}

fn anchor_from_match(
    id: String,
    track_id: &str,
    ref_time: f64,
    m: &crate::xcorr::LocalMatch,
    sample_rate: u32,
    expected: f64,
) -> Anchor {
    let candidates: Vec<crate::anchor::LagCandidate> = m
        .candidates
        .iter()
        .map(|p| crate::anchor::LagCandidate {
            lag_secs: (expected + p.lag_samples / sample_rate as f64) - ref_time,
            score: p.score,
        })
        .collect();
    let member_time = m.member_time;
    let ambiguous = candidates.len() > 1
        && candidates
            .get(1)
            .map(|c| m.score - c.score < 0.1)
            .unwrap_or(false);
    let status = if ambiguous && m.score < 0.9 {
        AnchorStatus::Unknown
    } else {
        AnchorStatus::Auto
    };
    Anchor {
        id,
        track_id: track_id.to_string(),
        ref_time,
        member_time,
        lag_secs: member_time - ref_time,
        status,
        candidates,
        confidence: m.score.max(0.0),
        residual_secs: None,
        note: None,
    }
}

fn unknown_anchor(id: String, track_id: &str, ref_time: f64) -> Anchor {
    Anchor {
        id,
        track_id: track_id.to_string(),
        ref_time,
        member_time: ref_time,
        lag_secs: 0.0,
        status: AnchorStatus::Unknown,
        candidates: Vec::new(),
        confidence: 0.0,
        residual_secs: None,
        note: Some("reference gap: no matching event".into()),
    }
}
