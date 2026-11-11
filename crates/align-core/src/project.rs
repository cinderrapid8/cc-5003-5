use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::anchor::Anchor;
use crate::error::{AlignError, Result};
use crate::map::TimeMap;

/// Source metadata for one imported member track.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackMeta {
    pub id: String,
    pub name: String,
    /// Path relative to the project file, when the project is saved alongside
    /// copied media; otherwise absolute.
    pub path: PathBuf,
    pub nominal_rate: u32,
    pub channels: u16,
    pub frames: usize,
    /// SHA-256 of the source file, so stale projects after media replacement
    /// are detected instead of silently reusing anchors.
    pub source_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectSettings {
    /// Output rate of the aligned project; all sources claim this nominal rate.
    pub project_rate: u32,
    /// Drift correction enabled. When false, maps use slope 1.0.
    pub correct_drift: bool,
    /// Windowed-sinc half taps used for the last render.
    pub kernel_taps_half: usize,
    /// Padding rule applied at output edges (documented in render reports).
    pub padding: PaddingRule,
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            project_rate: 48_000,
            correct_drift: true,
            kernel_taps_half: 32,
            padding: PaddingRule::ZeroFill,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaddingRule {
    /// Frames outside any mapped source sample are silence.
    ZeroFill,
}

/// Persistent alignment project.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    pub version: u32,
    pub name: String,
    pub tracks: Vec<TrackMeta>,
    pub reference_track_id: Option<String>,
    pub settings: ProjectSettings,
    /// Anchors keyed per member track. The reference itself has none.
    pub anchors: Vec<Anchor>,
    /// Fitted continuous time maps per member track.
    pub maps: Vec<TimeMap>,
    /// Bumped whenever the reference changes or source hashes change. Any
    /// derived cache tagged with an older epoch is invalid and must be
    /// recomputed; mixing old and new caches is rejected explicitly.
    pub epoch: u64,
    /// Silence/gap intervals on the reference timeline (start, end seconds).
    /// Anchors inside these are held Unknown rather than force-matched.
    pub reference_gaps: Vec<(f64, f64)>,
    /// Interruption intervals on each member's own timeline (start, end).
    pub member_gaps: std::collections::HashMap<String, Vec<(f64, f64)>>,
    /// Fit summaries (delay, drift ppm, residuals) per track, for the UI.
    pub fit_summaries: Vec<FitSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitSummary {
    pub track_id: String,
    pub delay_secs: f64,
    pub drift_ppm: f64,
    pub rms_residual_secs: f64,
    pub max_abs_residual_secs: f64,
    pub anchors_used: usize,
    pub anchors_rejected: usize,
    pub anchors_unknown: usize,
    pub known_fraction: f64,
}

impl Project {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            version: 1,
            name: name.into(),
            tracks: Vec::new(),
            reference_track_id: None,
            settings: ProjectSettings::default(),
            anchors: Vec::new(),
            maps: Vec::new(),
            epoch: 1,
            reference_gaps: Vec::new(),
            member_gaps: std::collections::HashMap::new(),
            fit_summaries: Vec::new(),
        }
    }

    pub fn add_track(&mut self, meta: TrackMeta) {
        if self.reference_track_id.is_none() {
            self.reference_track_id = Some(meta.id.clone());
            self.bump_epoch();
        }
        self.tracks.push(meta);
    }

    /// Switch the reference. All anchors and maps reference the old
    /// timeline, so they are cleared and the epoch is bumped; persisted caches
    /// carrying the old epoch will be rejected on load.
    pub fn set_reference(&mut self, track_id: &str) -> Result<()> {
        if !self.tracks.iter().any(|t| t.id == track_id) {
            return Err(AlignError::TrackNotFound(track_id.to_string()));
        }
        if self.reference_track_id.as_deref() == Some(track_id) {
            return Ok(());
        }
        self.reference_track_id = Some(track_id.to_string());
        self.anchors.clear();
        self.maps.clear();
        self.fit_summaries.clear();
        self.bump_epoch();
        Ok(())
    }

    pub fn bump_epoch(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Reject any derived artifact whose epoch does not match. This is the
    /// guard against reusing old anchors/caches after a reference switch.
    pub fn check_epoch(&self, artifact_epoch: u64) -> Result<()> {
        if artifact_epoch != self.epoch {
            Err(AlignError::StaleCache {
                found: artifact_epoch,
                expected: self.epoch,
            })
        } else {
            Ok(())
        }
    }

    pub fn member_tracks(&self) -> impl Iterator<Item = &TrackMeta> {
        let ref_id = self.reference_track_id.clone();
        self.tracks
            .iter()
            .filter(move |t| Some(&t.id) != ref_id.as_ref())
    }

    pub fn reference(&self) -> Result<&TrackMeta> {
        let id = self
            .reference_track_id
            .as_ref()
            .ok_or(AlignError::NoReference)?;
        self.tracks
            .iter()
            .find(|t| &t.id == id)
            .ok_or_else(|| AlignError::TrackNotFound(id.clone()))
    }

    pub fn anchors_for(&self, track_id: &str) -> Vec<&Anchor> {
        self.anchors.iter().filter(|a| a.track_id == track_id).collect()
    }

    pub fn anchors_for_mut(&mut self, track_id: &str) -> Vec<&mut Anchor> {
        self.anchors
            .iter_mut()
            .filter(|a| a.track_id == track_id)
            .collect()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let project: Project = serde_json::from_str(&json)?;
        Ok(project)
    }
}
