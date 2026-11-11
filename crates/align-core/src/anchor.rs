use serde::{Deserialize, Serialize};

/// Confidence lifecycle of an anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnchorStatus {
    /// Auto-detected and consistent with the fitted map.
    Auto,
    /// The user picked one of several correlation candidates.
    UserPicked,
    /// The user placed or corrected the anchor manually.
    Manual,
    /// Detected but rejected (outlier or ambiguous); kept for the audit trail.
    Rejected,
    /// No trustworthy match in this region. It is never used for fitting and
    /// the time map stays unknown there rather than being forced.
    Unknown,
}

/// A cross-correlation candidate for one anchor position.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LagCandidate {
    /// Lag in seconds: `ref(t) ~= member(t + lag)`.
    pub lag_secs: f64,
    /// Normalized correlation score in [-1, 1].
    pub score: f64,
}

/// One time correspondence between the reference timeline and a member track.
///
/// All times are continuous seconds on each track's own nominal-rate clock.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Anchor {
    pub id: String,
    pub track_id: String,
    /// Position on the reference timeline.
    pub ref_time: f64,
    /// Position on the member track's own timeline for the same physical event.
    pub member_time: f64,
    /// Lag in seconds (member_time - ref_time) at this anchor.
    pub lag_secs: f64,
    pub status: AnchorStatus,
    pub candidates: Vec<LagCandidate>,
    pub confidence: f64,
    /// Residual in seconds against the currently fitted time map.
    pub residual_secs: Option<f64>,
    pub note: Option<String>,
}

impl Anchor {
    pub fn pick_candidate(&mut self, index: usize) -> bool {
        if let Some(c) = self.candidates.get(index) {
            self.lag_secs = c.lag_secs;
            self.member_time = self.ref_time + c.lag_secs;
            self.status = AnchorStatus::UserPicked;
            self.confidence = c.score.max(0.0);
            self.residual_secs = None;
            true
        } else {
            false
        }
    }

    pub fn set_manual(&mut self, member_time: f64) {
        self.member_time = member_time;
        self.lag_secs = member_time - self.ref_time;
        self.status = AnchorStatus::Manual;
        self.residual_secs = None;
    }

    pub fn mark_unknown(&mut self) {
        self.status = AnchorStatus::Unknown;
        self.residual_secs = None;
    }

    pub fn is_usable(&self) -> bool {
        matches!(
            self.status,
            AnchorStatus::Auto | AnchorStatus::UserPicked | AnchorStatus::Manual
        )
    }
}
