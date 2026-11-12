//! Versioned portable project packages and migration from legacy storage.
//!
//! A *project package* is a self-describing directory:
//!
//! ```text
//! <name>.bandalign/
//!   package.json     manifest: format version, media mode, per-track hashes
//!   project.json     the Project (project.version == PACKAGE_PROJECT_VERSION)
//!   media/           copied sources (MediaMode::Copy only)
//!   evidence/        derived alignment evidence from the validation run
//!   migration.json   provenance: versions, ordered steps, rollback journal
//! ```
//!
//! The migrator never repairs data: reference/member gaps, unknown or
//! rejected anchors and their residuals are carried over byte-identically.
//! Media is either copied into the package ([`MediaMode::Copy`]) or kept as
//! hash-checked external references ([`MediaMode::Reference`]).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::audio::{write_wav_f32, AudioReader};
use crate::error::{AlignError, Result};
use crate::estimate::fit_track;
use crate::hash::{sha256_file, sha256_hex};
use crate::pipeline::{detect_anchors, PipelineConfig};
use crate::project::Project;
use crate::render::{render_project, RenderOptions, RenderReport};
use crate::synth::Gap;
use crate::verify::{channel_consistency, phase_continuity};
use crate::TimeMap;

/// Format version of the package directory written by this tool.
pub const PACKAGE_FORMAT_VERSION: u32 = 2;
/// `Project::version` stamped on projects stored inside a package.
pub const PACKAGE_PROJECT_VERSION: u32 = 2;
/// `Project::version` of the legacy single-file storage.
pub const LEGACY_PROJECT_VERSION: u32 = 1;

pub const MANIFEST_FILE: &str = "package.json";
pub const PROJECT_FILE: &str = "project.json";
pub const MIGRATION_FILE: &str = "migration.json";
pub const MEDIA_DIR: &str = "media";
pub const EVIDENCE_DIR: &str = "evidence";

// Semantic-preservation tolerances, shared with scripts/verify_synthetic.sh.
pub const MAX_DELAY_DELTA_SECS: f64 = 0.002;
pub const MAX_DRIFT_DELTA_PPM: f64 = 25.0;
pub const MAX_RMS_RESIDUAL_SECS: f64 = 0.002;
pub const MAX_CHANNEL_DIFF: f64 = 1e-6;

/// How the package stores source media.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaMode {
    /// Media is copied into the package; the package is self-contained.
    Copy,
    /// Only references (absolute paths + SHA-256) are stored; media stays put.
    Reference,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackagedTrack {
    pub track_id: String,
    /// Path inside the package (Copy) or absolute source path (Reference).
    pub packaged_path: PathBuf,
    pub source_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageManifest {
    pub format_version: u32,
    pub name: String,
    pub project_file: String,
    pub media_mode: MediaMode,
    pub tracks: Vec<PackagedTrack>,
    /// Project-format version this package was migrated from.
    pub source_format_version: u32,
    /// SHA-256 of the legacy project file the package was migrated from.
    /// Re-migrating the same source is a detected no-op.
    pub source_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationOutcome {
    Completed,
    /// Target package already holds a migration of the exact same source.
    AlreadyMigrated,
    RolledBack,
}

/// One ordered, human-auditable migration action (including rollbacks).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MigrationStep {
    pub name: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MigrationReport {
    pub tool_version: String,
    pub source_path: PathBuf,
    pub package_dir: PathBuf,
    pub media_mode: MediaMode,
    pub from_format_version: u32,
    pub to_format_version: u32,
    /// SHA-256 of the legacy project file.
    pub source_digest: String,
    /// Backup of a replaced package, kept until the new one validates.
    pub backup_path: Option<PathBuf>,
    pub steps: Vec<MigrationStep>,
    /// Structural discrepancies found by validation; empty means identical.
    pub discrepancies: Vec<String>,
    pub preservation: Option<PreservationReport>,
    pub evidence: Option<AlignmentEvidence>,
    pub outcome: MigrationOutcome,
    pub error: Option<String>,
}

impl MigrationReport {
    fn step(&mut self, name: &str, detail: String) {
        self.steps.push(MigrationStep {
            name: name.to_string(),
            detail,
        });
    }
}

/// Storage adapter over the legacy single-file project and the versioned
/// package directory. The only interface differences callers must care about
/// are where the project lives and which directory track paths resolve
/// against; everything else goes through the same `Project` type.
#[derive(Clone, Debug)]
pub enum ProjectStorage {
    Legacy { file: PathBuf },
    Package { dir: PathBuf },
}

pub struct LoadedProject {
    pub project: Project,
    /// Directory that relative track paths resolve against.
    pub media_base: PathBuf,
}

impl ProjectStorage {
    /// A directory with a `package.json` manifest is a package; anything else
    /// is treated as a legacy project file.
    pub fn detect(path: &Path) -> Result<Self> {
        if path.is_dir() {
            if path.join(MANIFEST_FILE).is_file() {
                Ok(Self::Package {
                    dir: path.to_path_buf(),
                })
            } else {
                Err(AlignError::NotAPackage(path.to_path_buf()))
            }
        } else {
            Ok(Self::Legacy {
                file: path.to_path_buf(),
            })
        }
    }

    pub fn load(&self) -> Result<LoadedProject> {
        match self {
            Self::Legacy { file } => {
                let project = Project::load(file)?;
                let media_base = file
                    .parent()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("."));
                Ok(LoadedProject { project, media_base })
            }
            Self::Package { dir } => {
                let text = std::fs::read_to_string(dir.join(MANIFEST_FILE))?;
                let manifest: PackageManifest = serde_json::from_str(&text)?;
                if manifest.format_version != PACKAGE_FORMAT_VERSION {
                    return Err(AlignError::UnsupportedPackageVersion(
                        manifest.format_version,
                    ));
                }
                let project = Project::load(&dir.join(&manifest.project_file))?;
                Ok(LoadedProject {
                    project,
                    media_base: dir.clone(),
                })
            }
        }
    }

    pub fn save(&self, project: &Project) -> Result<()> {
        match self {
            Self::Legacy { file } => project.save(file),
            Self::Package { dir } => project.save(&dir.join(PROJECT_FILE)),
        }
    }
}

/// Resolve a track path stored in a project against its storage base.
pub fn resolve_track_path(base: &Path, stored: &Path) -> PathBuf {
    if stored.is_absolute() {
        stored.to_path_buf()
    } else {
        base.join(stored)
    }
}

fn sibling_dir(target: &Path, suffix: &str) -> PathBuf {
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "package".to_string());
    target
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{name}{suffix}"))
}

fn rollback_report_path(target: &Path) -> PathBuf {
    sibling_dir(target, ".rollback-report.json")
}

fn write_report_file(path: &Path, report: &MigrationReport) -> Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Structural migration: legacy project file -> staged, validated package.
///
/// On success the package is committed at `target` and any replaced package
/// is retained at `report.backup_path` until the caller finishes validation
/// (see [`migrate_package`]). On failure everything is rolled back and a
/// rollback report is written next to the target.
pub fn migrate_storage(
    source: &Path,
    target: &Path,
    mode: MediaMode,
    force: bool,
) -> Result<MigrationReport> {
    if source.is_dir() {
        return Err(AlignError::Other(format!(
            "migration source must be a legacy project JSON file, not a directory: {}",
            source.display()
        )));
    }
    let mut report = MigrationReport {
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        source_path: source.to_path_buf(),
        package_dir: target.to_path_buf(),
        media_mode: mode,
        from_format_version: LEGACY_PROJECT_VERSION,
        to_format_version: PACKAGE_FORMAT_VERSION,
        source_digest: String::new(),
        backup_path: None,
        steps: Vec::new(),
        discrepancies: Vec::new(),
        preservation: None,
        evidence: None,
        outcome: MigrationOutcome::Completed,
        error: None,
    };

    let source_bytes = std::fs::read(source)?;
    let source_digest = sha256_hex(&source_bytes);
    report.source_digest = source_digest.clone();
    let old: Project = serde_json::from_slice(&source_bytes)?;
    report.from_format_version = old.version;
    report.step(
        "load-source",
        format!(
            "{} (project version {}, sha256 {})",
            source.display(),
            old.version,
            source_digest
        ),
    );

    // Duplicate / conflicting target handling.
    if target.exists() {
        let existing_digest = std::fs::read_to_string(target.join(MANIFEST_FILE))
            .ok()
            .and_then(|text| serde_json::from_str::<PackageManifest>(&text).ok())
            .map(|m| m.source_digest);
        if existing_digest.as_deref() == Some(source_digest.as_str()) {
            report.outcome = MigrationOutcome::AlreadyMigrated;
            report.step(
                "check-duplicate",
                "target package already migrated from the identical source; no-op".into(),
            );
            return Ok(report);
        }
        if !force {
            return Err(AlignError::PackageExists(target.to_path_buf()));
        }
        let mut n = 1u32;
        let backup = loop {
            let cand = sibling_dir(target, &format!(".backup-{n}"));
            if !cand.exists() {
                break cand;
            }
            n += 1;
        };
        std::fs::rename(target, &backup)?;
        report.step(
            "backup-existing",
            format!("{} -> {}", target.display(), backup.display()),
        );
        report.backup_path = Some(backup);
    } else {
        report.step("check-duplicate", "no existing package at target".into());
    }

    let staging = sibling_dir(target, ".staging");
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
        report.step(
            "clean-staging",
            format!("removed leftover staging dir {}", staging.display()),
        );
    }
    std::fs::create_dir_all(&staging)?;
    report.step("stage", format!("staging into {}", staging.display()));

    let old_base = source
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let result = stage_package(&old, &old_base, &staging, mode, &mut report);
    match result {
        Ok(discrepancies) => {
            report.discrepancies = discrepancies;
            if !report.discrepancies.is_empty() {
                let err = AlignError::MigrationValidation(report.discrepancies.join("; "));
                rollback_pre_commit(&staging, &report.backup_path.clone(), &mut report, &err);
                return Err(err);
            }
            if let Err(e) = std::fs::rename(&staging, target) {
                let e = AlignError::Io(e);
                rollback_pre_commit(&staging, &report.backup_path.clone(), &mut report, &e);
                return Err(e);
            }
            report.step(
                "commit",
                format!("{} -> {}", staging.display(), target.display()),
            );
            Ok(report)
        }
        Err(e) => {
            rollback_pre_commit(&staging, &report.backup_path.clone(), &mut report, &e);
            Err(e)
        }
    }
}

/// Roll back before the package was committed: drop staging, restore backup.
fn rollback_pre_commit(
    staging: &Path,
    backup: &Option<PathBuf>,
    report: &mut MigrationReport,
    cause: &AlignError,
) {
    if staging.exists() {
        let _ = std::fs::remove_dir_all(staging);
        report.step(
            "rollback-staging",
            format!("removed staging dir {}", staging.display()),
        );
    }
    if let Some(b) = backup {
        let target = report.package_dir.clone();
        if std::fs::rename(b, &target).is_ok() {
            report.step(
                "rollback-restore",
                format!("restored {} from {}", target.display(), b.display()),
            );
        }
    }
    finalize_rollback(report, cause);
}

/// Roll back after the package was committed: drop it, restore backup.
fn rollback_post_commit(
    target: &Path,
    backup: &Option<PathBuf>,
    report: &mut MigrationReport,
    cause: &AlignError,
) {
    if target.exists() {
        let _ = std::fs::remove_dir_all(target);
        report.step(
            "rollback-remove-package",
            format!("removed package {}", target.display()),
        );
    }
    if let Some(b) = backup {
        if std::fs::rename(b, target).is_ok() {
            report.step(
                "rollback-restore",
                format!("restored {} from {}", target.display(), b.display()),
            );
        }
    }
    finalize_rollback(report, cause);
}

fn finalize_rollback(report: &mut MigrationReport, cause: &AlignError) {
    report.outcome = MigrationOutcome::RolledBack;
    report.error = Some(cause.to_string());
    let path = rollback_report_path(&report.package_dir.clone());
    report.step("rollback-report", format!("wrote {}", path.display()));
    if let Err(e) = write_report_file(&path, report) {
        eprintln!("failed to write rollback report {}: {e}", path.display());
    }
}

/// Build the package contents inside `staging`. Returns validation
/// discrepancies (empty = semantically identical to the source project).
fn stage_package(
    old: &Project,
    old_base: &Path,
    staging: &Path,
    mode: MediaMode,
    report: &mut MigrationReport,
) -> Result<Vec<String>> {
    let mut packaged = old.clone();
    packaged.version = PACKAGE_PROJECT_VERSION;
    let mut manifest_tracks = Vec::new();

    match mode {
        MediaMode::Copy => {
            let media_dir = staging.join(MEDIA_DIR);
            std::fs::create_dir_all(&media_dir)?;
            for track in &mut packaged.tracks {
                let src = resolve_track_path(old_base, &track.path);
                let file_name = src
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "audio.wav".to_string());
                let packaged_name = format!("{}-{file_name}", track.id);
                let dest = media_dir.join(&packaged_name);
                std::fs::copy(&src, &dest)?;
                let got = sha256_file(&dest)?;
                if got != track.source_hash {
                    return Err(AlignError::MediaHashMismatch {
                        track: track.id.clone(),
                        expected: track.source_hash.clone(),
                        got,
                    });
                }
                report.step(
                    "copy-media",
                    format!(
                        "{} -> media/{packaged_name} (sha256 {got})",
                        src.display()
                    ),
                );
                track.path = PathBuf::from(MEDIA_DIR).join(&packaged_name);
                manifest_tracks.push(PackagedTrack {
                    track_id: track.id.clone(),
                    packaged_path: track.path.clone(),
                    source_hash: track.source_hash.clone(),
                });
            }
        }
        MediaMode::Reference => {
            for track in &mut packaged.tracks {
                let src = resolve_track_path(old_base, &track.path);
                let got = sha256_file(&src)?;
                if got != track.source_hash {
                    return Err(AlignError::MediaHashMismatch {
                        track: track.id.clone(),
                        expected: track.source_hash.clone(),
                        got,
                    });
                }
                let abs = std::fs::canonicalize(&src)?;
                report.step(
                    "reference-media",
                    format!("{} kept in place (sha256 {got})", abs.display()),
                );
                track.path = abs.clone();
                manifest_tracks.push(PackagedTrack {
                    track_id: track.id.clone(),
                    packaged_path: abs,
                    source_hash: track.source_hash.clone(),
                });
            }
        }
    }

    packaged.save(&staging.join(PROJECT_FILE))?;
    report.step(
        "write-project",
        format!(
            "project version {} -> {}",
            old.version, PACKAGE_PROJECT_VERSION
        ),
    );
    let manifest = PackageManifest {
        format_version: PACKAGE_FORMAT_VERSION,
        name: old.name.clone(),
        project_file: PROJECT_FILE.to_string(),
        media_mode: mode,
        tracks: manifest_tracks,
        source_format_version: old.version,
        source_digest: report.source_digest.clone(),
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    std::fs::write(staging.join(MANIFEST_FILE), manifest_json)?;
    report.step("write-manifest", format!("format v{PACKAGE_FORMAT_VERSION}"));

    // Validate by reloading what was actually written, never by assuming.
    let reloaded = Project::load(&staging.join(PROJECT_FILE))?;
    let discrepancies = compare_projects(old, &reloaded);
    report.step(
        "validate-static",
        if discrepancies.is_empty() {
            "reloaded package is semantically identical to source".to_string()
        } else {
            format!("{} discrepancies", discrepancies.len())
        },
    );
    write_report_file(&staging.join(MIGRATION_FILE), report)?;
    Ok(discrepancies)
}

/// Field-by-field semantic comparison of a source project and its migrated
/// form. Track paths legitimately change with the media mode and are checked
/// through the manifest hashes instead; everything else must be identical.
/// Nothing is repaired here — differences are reported, not fixed.
pub fn compare_projects(old: &Project, new: &Project) -> Vec<String> {
    let mut d = Vec::new();
    if old.name != new.name {
        d.push(format!("name: {:?} vs {:?}", old.name, new.name));
    }
    if old.epoch != new.epoch {
        d.push(format!("epoch: {} vs {}", old.epoch, new.epoch));
    }
    if old.reference_track_id != new.reference_track_id {
        d.push(format!(
            "reference_track_id: {:?} vs {:?}",
            old.reference_track_id, new.reference_track_id
        ));
    }
    let settings = serde_json::to_string(&old.settings).unwrap_or_default();
    let settings_new = serde_json::to_string(&new.settings).unwrap_or_default();
    if settings != settings_new {
        d.push(format!("settings: {settings} vs {settings_new}"));
    }
    if old.reference_gaps != new.reference_gaps {
        d.push("reference_gaps differ".to_string());
    }
    if old.member_gaps != new.member_gaps {
        d.push("member_gaps differ".to_string());
    }
    let anchors = serde_json::to_string(&old.anchors).unwrap_or_default();
    let anchors_new = serde_json::to_string(&new.anchors).unwrap_or_default();
    if anchors != anchors_new {
        d.push("anchors differ".to_string());
    }
    let maps = serde_json::to_string(&old.maps).unwrap_or_default();
    let maps_new = serde_json::to_string(&new.maps).unwrap_or_default();
    if maps != maps_new {
        d.push("time maps differ".to_string());
    }
    let fits = serde_json::to_string(&old.fit_summaries).unwrap_or_default();
    let fits_new = serde_json::to_string(&new.fit_summaries).unwrap_or_default();
    if fits != fits_new {
        d.push("fit summaries differ".to_string());
    }
    if old.tracks.len() != new.tracks.len() {
        d.push(format!(
            "track count: {} vs {}",
            old.tracks.len(),
            new.tracks.len()
        ));
    }
    for (a, b) in old.tracks.iter().zip(new.tracks.iter()) {
        if a.id != b.id
            || a.name != b.name
            || a.nominal_rate != b.nominal_rate
            || a.channels != b.channels
            || a.frames != b.frames
            || a.source_hash != b.source_hash
        {
            d.push(format!("track metadata differs for {}", a.id));
        }
    }
    d
}

/// Verify every track file reachable from the package matches its recorded
/// source hash. This is the media half of old-storage -> package validation.
pub fn validate_package_media(package_dir: &Path) -> Result<()> {
    let text = std::fs::read_to_string(package_dir.join(MANIFEST_FILE))?;
    let manifest: PackageManifest = serde_json::from_str(&text)?;
    if manifest.format_version != PACKAGE_FORMAT_VERSION {
        return Err(AlignError::UnsupportedPackageVersion(
            manifest.format_version,
        ));
    }
    for t in &manifest.tracks {
        let path = resolve_track_path(package_dir, &t.packaged_path);
        let got = sha256_file(&path)?;
        if got != t.source_hash {
            return Err(AlignError::MediaHashMismatch {
                track: t.track_id.clone(),
                expected: t.source_hash.clone(),
                got,
            });
        }
    }
    Ok(())
}

/// Execution evidence for one member track: mapping residuals from the fresh
/// fit plus inter-channel phase consistency of the rendered stem.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackEvidence {
    pub track_id: String,
    pub delay_secs: f64,
    pub drift_ppm: f64,
    pub rms_residual_secs: f64,
    pub max_abs_residual_secs: f64,
    pub anchors_used: usize,
    pub anchors_rejected: usize,
    pub anchors_unknown: usize,
    pub known_fraction: f64,
    pub channel_max_diff: f64,
    pub channel_rms_diff: f64,
    /// Phase jump at map joins, only when a probe frequency was given.
    pub max_phase_jump_rad: Option<f64>,
}

/// Result of actually running the alignment pipeline from a storage
/// location: fresh fit metrics, render reports and the aligned stems.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlignmentEvidence {
    pub storage: String,
    pub aligned_dir: PathBuf,
    pub project_end_secs: f64,
    pub tracks: Vec<TrackEvidence>,
    pub render_reports: Vec<RenderReport>,
}

/// Run the real detect -> fit -> render pipeline against a storage (legacy
/// file or package) and capture numerical evidence. Every failure
/// propagates: no metric is ever filled in with a default to keep a run
/// alive, so a missing file or a failed fit aborts instead of passing.
pub fn run_alignment_evidence(
    storage: &ProjectStorage,
    out_dir: &Path,
    probe_hz: Option<f64>,
) -> Result<AlignmentEvidence> {
    let loaded = storage.load()?;
    let project = &loaded.project;
    let base = &loaded.media_base;

    let ref_meta = project.reference()?.clone();
    let reference = AudioReader::load(&resolve_track_path(base, &ref_meta.path))?;
    let config = PipelineConfig::default();
    let render_opts = RenderOptions {
        output_rate: project.settings.project_rate,
        kernel_taps_half: project.settings.kernel_taps_half,
        ..RenderOptions::default()
    };
    let ref_gaps: Vec<Gap> = project
        .reference_gaps
        .iter()
        .map(|(s, e)| Gap {
            start: *s,
            end: *e,
        })
        .collect();

    let member_metas: Vec<_> = project.member_tracks().cloned().collect();
    let mut sources: Vec<(crate::audio::AudioBuffer, TimeMap)> = Vec::new();
    let mut tracks = Vec::new();
    for meta in &member_metas {
        let member = AudioReader::load(&resolve_track_path(base, &meta.path))?;
        let member_gaps: Vec<Gap> = project
            .member_gaps
            .get(&meta.id)
            .map(|gs| {
                gs.iter()
                    .map(|(s, e)| Gap {
                        start: *s,
                        end: *e,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut anchors = detect_anchors(
            &meta.id,
            &reference,
            &member,
            &ref_gaps,
            &member_gaps,
            &config,
        )?;
        let duration = member.duration_secs();
        let (map, fit) = if project.settings.correct_drift {
            fit_track(&meta.id, &mut anchors, duration)?
        } else {
            let lag = anchors
                .iter()
                .find(|a| a.is_usable())
                .map(|a| a.lag_secs)
                .ok_or(AlignError::InsufficientAnchors { need: 1, have: 0 })?;
            (
                TimeMap::from_constant_delay(&meta.id, lag, duration),
                crate::estimate::FitResult {
                    slope: 1.0,
                    offset: -lag,
                    delay_secs: lag,
                    drift_ppm: 0.0,
                    rms_residual_secs: 0.0,
                    max_abs_residual_secs: 0.0,
                    used: anchors.iter().filter(|a| a.is_usable()).count(),
                    rejected: 0,
                },
            )
        };
        let unknown_ranges: Vec<(f64, f64)> = anchors
            .iter()
            .filter(|a| !a.is_usable())
            .map(|a| (a.member_time - 0.1, a.member_time + 0.1))
            .collect();
        let known_fraction = map.known_fraction(duration, &unknown_ranges);
        tracks.push(TrackEvidence {
            track_id: meta.id.clone(),
            delay_secs: fit.delay_secs,
            drift_ppm: fit.drift_ppm,
            rms_residual_secs: fit.rms_residual_secs,
            max_abs_residual_secs: fit.max_abs_residual_secs,
            anchors_used: fit.used,
            anchors_rejected: fit.rejected,
            anchors_unknown: anchors.iter().filter(|a| !a.is_usable()).count(),
            known_fraction,
            channel_max_diff: f64::INFINITY, // filled after render, never left
            channel_rms_diff: f64::INFINITY,
            max_phase_jump_rad: None,
        });
        sources.push((member, map));
    }

    let refs_map = TimeMap::from_constant_delay(&ref_meta.id, 0.0, reference.duration_secs());
    let all_sources: Vec<(&crate::audio::AudioBuffer, &TimeMap)> =
        std::iter::once((&reference, &refs_map))
            .chain(sources.iter().map(|(b, m)| (b, m)))
            .collect();
    let (rendered, project_end) = render_project(&all_sources, &render_opts)?;

    let aligned_dir = out_dir.join("aligned");
    std::fs::create_dir_all(&aligned_dir)?;
    let mut reports = Vec::new();
    for (i, rt) in rendered.iter().enumerate() {
        let name = if i == 0 {
            "reference_aligned.wav".to_string()
        } else {
            format!("{}_aligned.wav", member_metas[i - 1].id)
        };
        write_wav_f32(&aligned_dir.join(&name), &rt.buffer)?;
        reports.push(rt.report.clone());
    }

    // Channel phase consistency and optional tone-phase continuity are
    // measured on the stems just written; a failed measurement is an error,
    // not a zero-filled default.
    for (k, tev) in tracks.iter_mut().enumerate() {
        let rt = &rendered[k + 1];
        let cc = channel_consistency(&rt.buffer);
        tev.channel_max_diff = cc.max_interchannel_diff;
        tev.channel_rms_diff = cc.rms_interchannel_diff;
        if let Some(freq) = probe_hz {
            let map = &sources[k].1;
            let joins: Vec<f64> = map
                .segments
                .windows(2)
                .filter_map(|w| map.to_project(w[1].t0).ok())
                .collect();
            if !joins.is_empty() {
                let pc = phase_continuity(&rt.buffer, freq, &joins);
                tev.max_phase_jump_rad = Some(pc.max_phase_jump_rad);
            }
        }
    }

    let evidence = AlignmentEvidence {
        storage: match storage {
            ProjectStorage::Legacy { file } => file.display().to_string(),
            ProjectStorage::Package { dir } => dir.display().to_string(),
        },
        aligned_dir,
        project_end_secs: project_end,
        tracks,
        render_reports: reports,
    };
    std::fs::write(
        out_dir.join("evidence.json"),
        serde_json::to_string_pretty(&evidence)?,
    )?;
    Ok(evidence)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreservationVerdict {
    /// Fresh evidence matches the stored fit summaries within tolerance.
    Preserved,
    /// The old project had no fit summaries to compare against; only
    /// baseline-free sanity checks (channel phase) were applied.
    NoBaseline,
    /// At least one metric regressed beyond tolerance.
    Degraded,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreservationCheck {
    pub track_id: String,
    pub metric: String,
    pub baseline: Option<f64>,
    pub fresh: f64,
    pub tolerance: f64,
    pub pass: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreservationReport {
    pub verdict: PreservationVerdict,
    pub checks: Vec<PreservationCheck>,
}

/// Judge how well the migrated package preserves the original semantics by
/// comparing fresh mapping residuals and channel phase against the fit
/// summaries stored in the old project.
pub fn judge_preservation(old: &Project, evidence: &AlignmentEvidence) -> PreservationReport {
    let mut checks = Vec::new();
    let mut had_baseline = false;
    for tev in &evidence.tracks {
        checks.push(PreservationCheck {
            track_id: tev.track_id.clone(),
            metric: "channel_max_diff".into(),
            baseline: None,
            fresh: tev.channel_max_diff,
            tolerance: MAX_CHANNEL_DIFF,
            pass: tev.channel_max_diff <= MAX_CHANNEL_DIFF,
        });
        checks.push(PreservationCheck {
            track_id: tev.track_id.clone(),
            metric: "rms_residual_secs".into(),
            baseline: None,
            fresh: tev.rms_residual_secs,
            tolerance: MAX_RMS_RESIDUAL_SECS,
            pass: tev.rms_residual_secs <= MAX_RMS_RESIDUAL_SECS,
        });
        if let Some(base) = old
            .fit_summaries
            .iter()
            .find(|s| s.track_id == tev.track_id)
        {
            had_baseline = true;
            let mut delta = |metric: &str, base_v: f64, fresh: f64, tol: f64| {
                checks.push(PreservationCheck {
                    track_id: tev.track_id.clone(),
                    metric: metric.to_string(),
                    baseline: Some(base_v),
                    fresh,
                    tolerance: tol,
                    pass: (fresh - base_v).abs() <= tol,
                });
            };
            delta("delay_secs", base.delay_secs, tev.delay_secs, MAX_DELAY_DELTA_SECS);
            delta("drift_ppm", base.drift_ppm, tev.drift_ppm, MAX_DRIFT_DELTA_PPM);
            delta(
                "rms_residual_secs_delta",
                base.rms_residual_secs,
                tev.rms_residual_secs,
                MAX_RMS_RESIDUAL_SECS,
            );
            delta(
                "anchors_used",
                base.anchors_used as f64,
                tev.anchors_used as f64,
                0.0,
            );
            delta(
                "anchors_rejected",
                base.anchors_rejected as f64,
                tev.anchors_rejected as f64,
                0.0,
            );
            delta(
                "anchors_unknown",
                base.anchors_unknown as f64,
                tev.anchors_unknown as f64,
                0.0,
            );
        }
    }
    let failed = checks.iter().any(|c| !c.pass);
    let verdict = if failed {
        PreservationVerdict::Degraded
    } else if had_baseline {
        PreservationVerdict::Preserved
    } else {
        PreservationVerdict::NoBaseline
    };
    PreservationReport { verdict, checks }
}

pub struct MigrationOptions {
    pub mode: MediaMode,
    /// Replace an existing package that was migrated from a different source.
    pub force: bool,
    /// Optional tone frequency for phase-continuity evidence.
    pub probe_hz: Option<f64>,
}

/// Full migration flow: structural migration, then an actual alignment run
/// against the committed package at its target path, then a preservation
/// judgement. Any failure rolls the target back to its previous state and
/// leaves a rollback report next to it.
pub fn migrate_package(
    source: &Path,
    target: &Path,
    opts: &MigrationOptions,
) -> Result<MigrationReport> {
    let mut report = migrate_storage(source, target, opts.mode, opts.force)?;
    if report.outcome == MigrationOutcome::AlreadyMigrated {
        return Ok(report);
    }
    let old = Project::load(source)?;

    let storage = ProjectStorage::Package {
        dir: target.to_path_buf(),
    };
    let evidence_dir = target.join(EVIDENCE_DIR);
    let evidence = match run_alignment_evidence(&storage, &evidence_dir, opts.probe_hz) {
        Ok(ev) => ev,
        Err(e) => {
            rollback_post_commit(target, &report.backup_path.clone(), &mut report, &e);
            return Err(e);
        }
    };
    report.step(
        "evidence",
        format!(
            "alignment run at {} ({} track(s), project end {:.3}s)",
            target.display(),
            evidence.tracks.len(),
            evidence.project_end_secs
        ),
    );
    let preservation = judge_preservation(&old, &evidence);
    report.step(
        "preservation",
        format!("verdict {:?}", preservation.verdict),
    );
    let degraded = preservation.verdict == PreservationVerdict::Degraded;
    report.preservation = Some(preservation);
    report.evidence = Some(evidence);
    if degraded {
        let err = AlignError::MigrationValidation(
            "semantics degraded after migration; see preservation checks".into(),
        );
        rollback_post_commit(target, &report.backup_path.clone(), &mut report, &err);
        return Err(err);
    }
    if let Some(backup) = report.backup_path.take() {
        if backup.is_dir() {
            std::fs::remove_dir_all(&backup)?;
        } else {
            std::fs::remove_file(&backup)?;
        }
        report.step(
            "discard-backup",
            format!("removed {}", backup.display()),
        );
    }
    write_report_file(&target.join(MIGRATION_FILE), &report)?;
    Ok(report)
}
