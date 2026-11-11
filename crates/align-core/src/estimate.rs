use crate::anchor::{Anchor, AnchorStatus};
use crate::error::{AlignError, Result};
use crate::map::{MapSegment, TimeMap};

/// Fit result for one usable set of anchors.
#[derive(Clone, Debug)]
pub struct FitResult {
    /// slope = 1 + drift
    pub slope: f64,
    /// intercept of project_time = slope * member_time + offset
    pub offset: f64,
    pub delay_secs: f64,
    /// fractional clock drift (slope - 1), e.g. +50e-6 means 50 ppm fast.
    pub drift_ppm: f64,
    pub rms_residual_secs: f64,
    pub max_abs_residual_secs: f64,
    pub used: usize,
    pub rejected: usize,
}

struct Point {
    x: f64, // member time
    y: f64, // project (reference) time
    w: f64, // confidence weight
}

/// Robustly fit project_time = slope * member_time + offset from usable
/// anchors. Outliers are first removed with a small RANSAC sweep, then an
/// iteratively reweighted least squares pass polishes the fit. Rejected
/// anchors get `AnchorStatus::Rejected` and their residuals are recorded.
pub fn fit_track(
    track_id: &str,
    anchors: &mut [Anchor],
    member_duration: f64,
) -> Result<(TimeMap, FitResult)> {
    let usable_idx: Vec<usize> = anchors
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_usable() && a.track_id == track_id)
        .map(|(i, _)| i)
        .collect();

    if usable_idx.is_empty() {
        return Err(AlignError::InsufficientAnchors { need: 1, have: 0 });
    }

    // Single anchor: delay only, drift cannot be identified from one point.
    if usable_idx.len() == 1 {
        let a = &anchors[usable_idx[0]];
        let delay = a.lag_secs;
        let map = TimeMap::from_constant_delay(track_id, delay, member_duration);
        anchors[usable_idx[0]].residual_secs = Some(0.0);
        return Ok((
            map,
            FitResult {
                slope: 1.0,
                offset: -delay,
                delay_secs: delay,
                drift_ppm: 0.0,
                rms_residual_secs: 0.0,
                max_abs_residual_secs: 0.0,
                used: 1,
                rejected: 0,
            },
        ));
    }

    let points = |idx: &[usize]| -> Vec<Point> {
        idx.iter()
            .map(|&i| Point {
                x: anchors[i].member_time,
                y: anchors[i].ref_time,
                w: anchors[i].confidence.max(1e-3),
            })
            .collect()
    };

    // RANSAC-lite: all pairs define candidate lines; pick the one with the
    // most inliers. The 2 ms tolerance rejects gross mismatches (wrong
    // repeated-beat peak) while keeping sub-sample jitter of good anchors.
    let all = points(&usable_idx);
    let tol = 0.002;
    let mut best_inliers: Vec<usize> = Vec::new();
    let n = all.len();
    if n >= 2 {
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = all[j].x - all[i].x;
                if dx.abs() < 1e-6 {
                    continue;
                }
                let slope = (all[j].y - all[i].y) / dx;
                let offset = all[i].y - slope * all[i].x;
                let inliers: Vec<usize> = all
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| (p.y - (slope * p.x + offset)).abs() <= tol)
                    .map(|(k, _)| k)
                    .collect();
                if inliers.len() > best_inliers.len() {
                    best_inliers = inliers;
                }
            }
        }
    }
    if best_inliers.len() < 2 {
        best_inliers = (0..n).collect();
    }

    // IRLS on the inlier set with Huber-style bisquare weights.
    let mut active: Vec<usize> = best_inliers;
    let mut slope = 1.0;
    let mut offset = 0.0;
    for _ in 0..8 {
        let pts: Vec<Point> = active.iter().map(|&k| all[k].clone_with_w()).collect();
        let (s, b) = weighted_linfit(&pts);
        slope = s;
        offset = b;
        let resid: Vec<f64> = active.iter().map(|&k| all[k].y - (s * all[k].x + b)).collect();
        let sigma = 1.4826 * median_abs(&resid).max(1e-6);
        let cutoff = 4.685 * sigma;
        let next: Vec<usize> = active
            .iter()
            .zip(resid.iter())
            .filter(|(_, r)| r.abs() <= cutoff)
            .map(|(k, _)| *k)
            .collect();
        if next.len() == active.len() {
            break;
        }
        active = next;
    }

    let active_anchor_idx: Vec<usize> = active.iter().map(|k| usable_idx[*k]).collect();
    let rejected: Vec<usize> = usable_idx
        .iter()
        .filter(|i| !active_anchor_idx.contains(i))
        .copied()
        .collect();
    for &i in &rejected {
        anchors[i].status = AnchorStatus::Rejected;
        anchors[i].residual_secs = Some(anchors[i].ref_time - (slope * anchors[i].member_time + offset));
    }

    // Residuals for used anchors.
    let mut sq_sum = 0.0f64;
    let mut max_abs = 0.0f64;
    for &i in &active_anchor_idx {
        let r = anchors[i].ref_time - (slope * anchors[i].member_time + offset);
        anchors[i].residual_secs = Some(r);
        sq_sum += r * r;
        max_abs = max_abs.max(r.abs());
    }

    let delay = -offset; // project_time = member_time - delay at slope 1
    let map = TimeMap {
        track_id: track_id.to_string(),
        segments: vec![MapSegment {
            t0: 0.0,
            t1: member_duration,
            slope,
            offset,
            project_t0: offset,
            known: true,
        }],
    };

    Ok((
        map,
        FitResult {
            slope,
            offset,
            delay_secs: delay,
            drift_ppm: (slope - 1.0) * 1e6,
            rms_residual_secs: (sq_sum / active_anchor_idx.len() as f64).sqrt(),
            max_abs_residual_secs: max_abs,
            used: active_anchor_idx.len(),
            rejected: rejected.len(),
        },
    ))
}

/// Fit a segment that must start at `(t0, project_t0)` (continuity with the
/// previous segment). Only the slope is free, so joins are exact even with
/// few anchors in the segment.
pub fn fit_segment_continuous(
    t0: f64,
    project_t0: f64,
    t1: f64,
    anchors: &[Anchor],
) -> MapSegment {
    let pts: Vec<Point> = anchors
        .iter()
        .filter(|a| a.is_usable() && a.member_time >= t0 && a.member_time <= t1)
        .map(|a| Point {
            x: a.member_time - t0,
            y: a.ref_time - project_t0,
            w: a.confidence.max(1e-3),
        })
        .collect();
    // Minimize sum w (y - slope*x)^2 => slope = sum wxy / sum wx2.
    let mut num = 0.0;
    let mut den = 0.0;
    for p in &pts {
        num += p.w * p.x * p.y;
        den += p.w * p.x * p.x;
    }
    let slope = if den > 1e-12 { num / den } else { 1.0 };
    let offset = project_t0 - slope * t0;
    MapSegment {
        t0,
        t1,
        slope,
        offset,
        project_t0,
        known: !pts.is_empty(),
    }
}

/// Build a segmented map across recording gaps. `boundaries` are member
/// timestamps where a new segment starts (e.g. after an interruption); each
/// segment is fitted with continuity enforced at its start.
pub fn fit_segmented(
    track_id: &str,
    anchors: &mut [Anchor],
    boundaries: &[f64],
    member_duration: f64,
) -> Result<TimeMap> {
    let mut edges = vec![0.0];
    edges.extend_from_slice(boundaries);
    edges.push(member_duration);
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    edges.dedup();

    let mut segments = Vec::new();
    let mut project_cursor = 0.0;
    for w in edges.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        if t1 - t0 < 1e-9 {
            continue;
        }
        let seg = fit_segment_continuous(t0, project_cursor, t1, anchors);
        project_cursor = seg.project_time(t1);
        segments.push(seg);
    }
    Ok(TimeMap {
        track_id: track_id.to_string(),
        segments,
    })
}

fn weighted_linfit(p: &[Point]) -> (f64, f64) {
    let mut sw = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for q in p {
        let w = q.w;
        sw += w;
        sx += w * q.x;
        sy += w * q.y;
        sxx += w * q.x * q.x;
        sxy += w * q.x * q.y;
    }
    let det = sw * sxx - sx * sx;
    if det.abs() < 1e-12 {
        return (1.0, sy / sw - sx / sw);
    }
    let slope = (sw * sxy - sx * sy) / det;
    let offset = (sy - slope * sx) / sw;
    (slope, offset)
}

fn median_abs(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut a: Vec<f64> = x.iter().map(|v| v.abs()).collect();
    a.sort_by(|u, v| u.partial_cmp(v).unwrap_or(std::cmp::Ordering::Equal));
    a[a.len() / 2]
}

impl Point {
    fn clone_with_w(&self) -> Self {
        Point {
            x: self.x,
            y: self.y,
            w: self.w,
        }
    }
}
