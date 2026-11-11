use serde::{Deserialize, Serialize};

use crate::error::Result;

/// One continuous piece of a piecewise-linear time map.
///
/// Mapping convention: `project_time = slope * member_time + offset`,
/// where both times are in seconds. `slope = 1 + drift` captures the sample
/// clock ratio; `offset` captures the fixed delay. Boundaries between
/// segments share endpoints, so the overall map is continuous even when the
/// slope changes (e.g. across a recording interruption).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapSegment {
    /// Start of the segment on the member timeline (seconds).
    pub t0: f64,
    /// End of the segment on the member timeline (seconds).
    pub t1: f64,
    pub slope: f64,
    pub offset: f64,
    /// Project time at t0, stored explicitly so joins remain exact after
    /// serialization round-trips.
    pub project_t0: f64,
    /// False when this region has no trustworthy anchors. The segment is
    /// extrapolated from the nearest known slope for preview only and is
    /// flagged in every render report.
    pub known: bool,
}

impl MapSegment {
    pub fn project_time(&self, member_time: f64) -> f64 {
        self.slope * member_time + self.offset
    }
}

/// Piecewise-linear, non-decreasing mapping for one member track.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TimeMap {
    pub track_id: String,
    pub segments: Vec<MapSegment>,
}

/// Merge overlapping/adjacent (start, end) ranges.
pub fn merge_ranges(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = input
        .iter()
        .filter(|(a, b)| b > a)
        .copied()
        .collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out: Vec<(f64, f64)> = Vec::new();
    for r in v {
        if let Some(last) = out.last_mut() {
            if r.0 <= last.1 {
                last.1 = last.1.max(r.1);
                continue;
            }
        }
        out.push(r);
    }
    out
}

impl TimeMap {
    pub fn new(track_id: impl Into<String>) -> Self {
        Self {
            track_id: track_id.into(),
            segments: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn known_fraction(&self, member_duration: f64, unknown_ranges: &[(f64, f64)]) -> f64 {
        if member_duration <= 0.0 {
            return 0.0;
        }
        let merged = merge_ranges(unknown_ranges);
        let mut unknown = 0.0f64;
        for (a, b) in merged {
            let lo = a.clamp(0.0, member_duration);
            let hi = b.clamp(0.0, member_duration);
            unknown += (hi - lo).max(0.0);
        }
        let total_known = (member_duration - unknown).max(0.0);
        let mapped_known: f64 = self
            .segments
            .iter()
            .filter(|s| s.known)
            .map(|s| (s.t1 - s.t0).max(0.0))
            .sum();
        (mapped_known.min(total_known) / member_duration).clamp(0.0, 1.0)
    }

    fn locate(&self, member_time: f64) -> Option<&MapSegment> {
        self.segments
            .iter()
            .find(|s| member_time >= s.t0 && member_time <= s.t1)
            .or_else(|| self.segments.first())
    }

    /// Map a member-timeline time to the shared project timeline.
    pub fn to_project(&self, member_time: f64) -> Result<f64> {
        let s = self
            .locate(member_time)
            .ok_or(crate::error::AlignError::Other("empty time map".into()))?;
        Ok(s.project_time(member_time))
    }

    /// Inverse lookup: project time back to the member timeline.
    pub fn to_member(&self, project_time: f64) -> Result<f64> {
        let s = self
            .segments
            .iter()
            .find(|s| {
                let p0 = s.project_time(s.t0);
                let p1 = s.project_time(s.t1);
                project_time >= p0.min(p1) && project_time <= p0.max(p1)
            })
            .or_else(|| self.segments.first())
            .ok_or(crate::error::AlignError::Other("empty time map".into()))?;
        if s.slope.abs() < 1e-12 {
            return Ok(s.t0);
        }
        Ok((project_time - s.offset) / s.slope)
    }

    pub fn is_known_at(&self, member_time: f64) -> bool {
        self.locate(member_time).map(|s| s.known).unwrap_or(false)
    }

    /// Project-time span covered by the map.
    pub fn project_span(&self) -> Option<(f64, f64)> {
        let first = self.segments.first()?;
        let last = self.segments.last()?;
        Some((
            first.project_time(first.t0),
            last.project_time(last.t1),
        ))
    }

    /// Fixed-delay-only identity-slope map (used when the user disables drift
    /// correction or there is a single anchor).
    pub fn from_constant_delay(track_id: &str, delay: f64, duration: f64) -> Self {
        let seg = MapSegment {
            t0: 0.0,
            t1: duration,
            slope: 1.0,
            offset: -delay,
            project_t0: -delay,
            known: true,
        };        Self {
            track_id: track_id.to_string(),
            segments: vec![seg],
        }
    }
}
