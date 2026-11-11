use thiserror::Error;

#[derive(Debug, Error)]
pub enum AlignError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("wav decode error: {0}")]
    Wav(#[from] hound::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("track not found: {0}")]
    TrackNotFound(String),
    #[error("anchor not found: {0}")]
    AnchorNotFound(String),
    #[error("insufficient anchors: need {need}, have {have}")]
    InsufficientAnchors { need: usize, have: usize },
    #[error("anchor span too short: {0:.3}s")]
    SpanTooShort(f64),
    #[error("no reference track set")]
    NoReference,
    #[error("stale cache: epoch {found} does not match {expected}; recompute required")]
    StaleCache { found: u64, expected: u64 },
    #[error("incompatible sample rates: {0} vs {1}")]
    IncompatibleRate(u32, u32),
    #[error("render position outside source: t={t:.6}s")]
    OutsideSource { t: f64 },
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AlignError>;
