//! Integration tests for the alignment core using fully synthetic material
//! with known delay and drift, so every numerical claim is reproducible.

use align_core::anchor::AnchorStatus;
use align_core::estimate::fit_track;
use align_core::map::TimeMap;
use align_core::pipeline::{detect_anchors, PipelineConfig};
use align_core::render::{render_project, RenderOptions};
use align_core::synth::{
    render_member, render_reference, ClockModel, Gap, Hit, HitKind,
};
use align_core::verify::{
    channel_consistency, estimator_error, phase_continuity, GroundTruth,
};

fn hits(duration: f64) -> Vec<Hit> {
    let mut v = Vec::new();
    let mut t = 2.0;
    while t < duration - 1.0 {
        v.push(Hit { time: t, kind: HitKind::Clap });
        t += 4.0;
    }
    v
}

fn build_pair(
    duration: f64,
    delay: f64,
    drift_ppm: f64,
    channels: u16,
    ref_gaps: Vec<Gap>,
    member_gaps: Vec<Gap>,
) -> (align_core::audio::AudioBuffer, align_core::audio::AudioBuffer, ClockModel) {
    let rate = 48_000;
    let h = hits(duration);
    let tone = Some((440.0f32, 1.0, duration - 1.0));
    let reference = render_reference(&h, tone, &ref_gaps, duration, rate, channels);
    let model = ClockModel {
        delay_secs: delay,
        rate_ratio: 1.0 + drift_ppm * 1e-6,
        nominal_rate: rate,
    };
    let member = render_member(&h, tone, &model, &member_gaps, duration, channels);
    (reference, member, model)
}

#[test]
fn recovers_known_delay_and_drift_on_long_track() {
    let duration = 120.0;
    let (reference, member, model) =
        build_pair(duration, 0.2, 120.0, 2, vec![], vec![]);
    let mut anchors = detect_anchors(
        "m",
        &reference,
        &member,
        &[],
        &[],
        &PipelineConfig::default(),
    )
    .unwrap();
    let usable: Vec<_> = anchors.iter().filter(|a| a.is_usable()).collect();
    assert!(usable.len() >= 20, "expected many usable anchors, got {}", usable.len());
    // Anchors must span the full recording, not just the start.
    let span = usable.last().unwrap().ref_time - usable.first().unwrap().ref_time;
    assert!(span > duration * 0.8, "anchor span only {span}s");

    let (map, _fit) = fit_track("m", &mut anchors, member.duration_secs()).unwrap();
    let truth = GroundTruth {
        delay_secs: model.delay_secs,
        rate_ratio: model.rate_ratio,
    };
    let err = estimator_error(&map, &truth, member.duration_secs());
    assert!(err.delay_error_secs < 0.005, "delay error {:?}", err);
    assert!(err.drift_error_ppm < 25.0, "drift error {:?}", err);
    assert!(err.rms_residual_secs < 0.003, "rms {:?}", err);
}

#[test]
fn single_anchor_gives_delay_only_map() {
    let mut anchors = vec![align_core::anchor::Anchor {
        id: "x".into(),
        track_id: "m".into(),
        ref_time: 5.0,
        member_time: 5.1,
        lag_secs: 0.1,
        status: AnchorStatus::Manual,
        candidates: vec![],
        confidence: 1.0,
        residual_secs: None,
        note: None,
    }];
    let (map, fit) = fit_track("m", &mut anchors, 10.0).unwrap();
    assert!((fit.slope - 1.0).abs() < 1e-12);
    assert!((fit.delay_secs - 0.1).abs() < 1e-12);
    assert!((map.to_project(5.1).unwrap() - 5.0).abs() < 1e-9);
    // Round trip.
    let back = map.to_member(5.0).unwrap();
    assert!((back - 5.1).abs() < 1e-9);
}

#[test]
fn low_confidence_gap_regions_stay_unknown() {
    let duration = 60.0;
    let ref_gaps = vec![Gap { start: 20.0, end: 24.0 }];
    let member_gaps = vec![Gap { start: 40.0, end: 42.0 }];
    let (reference, member, _) =
        build_pair(duration, 0.08, 50.0, 1, ref_gaps.clone(), member_gaps.clone());
    let anchors = detect_anchors(
        "m",
        &reference,
        &member,
        &ref_gaps,
        &member_gaps,
        &PipelineConfig::default(),
    )
    .unwrap();
    let gap_unknown = anchors
        .iter()
        .filter(|a| a.status == AnchorStatus::Unknown)
        .count();
    assert!(gap_unknown >= 2, "gap regions must stay unknown: {gap_unknown}");
    // Unknown anchors must never be force-fit: they carry no residual.
    for a in &anchors {
        if a.status == AnchorStatus::Unknown {
            assert!(a.residual_secs.is_none());
        }
    }
}

#[test]
fn render_preserves_channels_and_common_length() {
    let duration = 60.0;
    let (reference, member, _model) = build_pair(duration, 0.1, 60.0, 2, vec![], vec![]);
    let mut anchors = detect_anchors(
        "m",
        &reference,
        &member,
        &[],
        &[],
        &PipelineConfig::default(),
    )
    .unwrap();
    let (map, _) = fit_track("m", &mut anchors, member.duration_secs()).unwrap();
    let ref_map = TimeMap::from_constant_delay("ref", 0.0, reference.duration_secs());
    let (tracks, end) =
        render_project(&[(&reference, &ref_map), (&member, &map)], &RenderOptions::default())
            .unwrap();
    // Every stem shares one project length.
    let len = tracks[0].buffer.frames();
    for t in &tracks {
        assert_eq!(t.buffer.frames(), len);
        assert_eq!(t.buffer.channels, 2);
    }
    assert!((end - duration).abs() < 0.05);
    // Identical source channels stay identical after resampling: one map, one
    // kernel, no per-channel timing divergence.
    let cc = channel_consistency(&tracks[1].buffer);
    assert!(cc.max_interchannel_diff < 1e-6, "channel divergence {cc:?}");
}

#[test]
fn phase_continuous_under_continuous_map() {
    let duration = 30.0;
    let (reference, member, _) = build_pair(duration, 0.05, 40.0, 1, vec![], vec![]);
    let mut anchors = detect_anchors(
        "m",
        &reference,
        &member,
        &[],
        &[],
        &PipelineConfig::default(),
    )
    .unwrap();
    let (map, _) = fit_track("m", &mut anchors, member.duration_secs()).unwrap();
    let rendered = render_project(
        &[(&member, &map)],
        &RenderOptions::default(),
    )
    .unwrap()
    .0
    .remove(0);
    // No segment joins in a single-segment map: measured jump must be zero.
    let report = phase_continuity(&rendered.buffer, 440.0, &[]);
    assert_eq!(report.max_phase_jump_rad, 0.0);
}

#[test]
fn segmented_map_is_continuous_at_joins() {
    use align_core::estimate::fit_segmented;
    let duration = 40.0;
    let (reference, member, _) = build_pair(duration, 0.1, 80.0, 1, vec![], vec![]);
    let mut anchors = detect_anchors(
        "m",
        &reference,
        &member,
        &[],
        &[],
        &PipelineConfig::default(),
    )
    .unwrap();
    let boundary = 20.0;
    let map = fit_segmented("m", &mut anchors, &[boundary], member.duration_secs()).unwrap();
    assert!(map.segments.len() >= 2);
    for w in map.segments.windows(2) {
        let left_end = w[0].project_time(w[0].t1);
        let right_start = w[1].project_time(w[1].t0);
        assert!(
            (left_end - right_start).abs() < 1e-9,
            "segment join discontinuity"
        );
    }
}

#[test]
fn interruption_unknown_region_does_not_corrupt_other_anchors() {
    let duration = 60.0;
    let member_gaps = vec![Gap { start: 30.0, end: 32.0 }];
    let (reference, member, model) =
        build_pair(duration, 0.1, 50.0, 1, vec![], member_gaps.clone());
    let anchors = detect_anchors(
        "m",
        &reference,
        &member,
        &[],
        &member_gaps,
        &PipelineConfig::default(),
    )
    .unwrap();
    // Anchors well before and after the interruption remain usable and
    // accurate; nothing near the gap is force-matched.
    let good: Vec<_> = anchors
        .iter()
        .filter(|a| a.is_usable() && (a.ref_time < 25.0 || a.ref_time > 35.0))
        .collect();
    assert!(good.len() >= 6);
    for a in &good {
        let true_lag = a.ref_time * (model.rate_ratio - 1.0) + model.delay_secs * model.rate_ratio;
        assert!(
            (a.lag_secs - true_lag).abs() < 0.005,
            "anchor {} off by {}",
            a.id,
            a.lag_secs - true_lag
        );
    }
    let near_gap: Vec<_> = anchors
        .iter()
        .filter(|a| (a.ref_time - 30.0).abs() < 3.0)
        .collect();
    assert!(
        near_gap.iter().any(|a| a.status == AnchorStatus::Unknown),
        "interruption region must contain an unknown anchor"
    );
}

#[test]
fn project_epoch_blocks_stale_cache_after_reference_switch() {
    use align_core::project::{Project, TrackMeta};
    use std::path::PathBuf;
    let mut p = Project::new("t");
    p.add_track(TrackMeta {
        id: "a".into(),
        name: "a".into(),
        path: PathBuf::from("a.wav"),
        nominal_rate: 48_000,
        channels: 1,
        frames: 100,
        source_hash: "h".into(),
    });
    p.add_track(TrackMeta {
        id: "b".into(),
        name: "b".into(),
        path: PathBuf::from("b.wav"),
        nominal_rate: 48_000,
        channels: 1,
        frames: 100,
        source_hash: "h".into(),
    });
    let old_epoch = p.epoch;
    p.set_reference("b").unwrap();
    assert!(p.check_epoch(old_epoch).is_err());
    assert!(p.check_epoch(p.epoch).is_ok());
    assert!(p.anchors.is_empty() && p.maps.is_empty());
}

#[test]
fn time_map_roundtrip_is_inverse() {
    let map = TimeMap::from_constant_delay("m", 0.123, 10.0);
    for t in [0.1, 1.5, 4.7, 9.9] {
        let p = map.to_project(t).unwrap();
        let back = map.to_member(p).unwrap();
        assert!((back - t).abs() < 1e-9);
    }
}

#[test]
fn segmented_render_has_no_phase_jump_at_join() {
    use align_core::estimate::fit_segmented;
    use align_core::render::{render_track, RenderOptions};
    let duration = 40.0;
    let (_, member, _) = build_pair(duration, 0.1, 80.0, 1, vec![], vec![]);
    let reference = render_reference(
        &hits(duration),
        None,
        &[],
        duration,
        48_000,
        1,
    );
    let mut anchors = detect_anchors(
        "m",
        &reference,
        &member,
        &[],
        &[],
        &PipelineConfig::default(),
    )
    .unwrap();
    let boundary = 20.0;
    let map = fit_segmented("m", &mut anchors, &[boundary], member.duration_secs()).unwrap();
    let rendered =
        render_track(&member, &map, Some(duration), &RenderOptions::default()).unwrap();
    let report = phase_continuity(&rendered.buffer, 440.0, &[boundary]);
    // A continuous piecewise map must not create a discontinuity larger than
    // the windowed-sinc interpolation noise at the segment join.
    assert!(
        report.max_phase_jump_rad < 0.5,
        "phase jump at join: {report:?}"
    );
}
