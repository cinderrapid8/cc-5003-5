//! Integration tests for legacy-project -> package migration. All material
//! is synthetic with known delay/drift so every assertion is reproducible.

use align_core::anchor::{Anchor, AnchorStatus};
use align_core::audio::{write_wav_f32, AudioBuffer};
use align_core::estimate::fit_track;
use align_core::hash::sha256_file;
use align_core::package::{
    compare_projects, judge_preservation, migrate_package, migrate_storage,
    run_alignment_evidence, MediaMode, MigrationOptions, MigrationOutcome,
    PackageManifest, PreservationVerdict, ProjectStorage, EVIDENCE_DIR,
    MANIFEST_FILE, MEDIA_DIR, MIGRATION_FILE, PACKAGE_FORMAT_VERSION,
    PACKAGE_PROJECT_VERSION, PROJECT_FILE,
};
use align_core::pipeline::{detect_anchors, PipelineConfig};
use align_core::project::{FitSummary, Project};
use align_core::synth::{
    render_member, render_reference, ClockModel, Gap, Hit, HitKind,
};
use align_core::AlignError;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "align-migrate-{tag}-{}-{n}",
        std::process::id()
    ));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn hits(duration: f64) -> Vec<Hit> {
    let mut v = Vec::new();
    let mut t = 2.0;
    while t < duration - 1.0 {
        v.push(Hit {
            time: t,
            kind: HitKind::Clap,
        });
        t += 4.0;
    }
    v
}

const RATE: u32 = 48_000;
const DURATION: f64 = 30.0;
const DELAY: f64 = 0.12;
const DRIFT_PPM: f64 = 75.0;

struct LegacyFixture {
    project_path: PathBuf,
    project: Project,
    member: AudioBuffer,
}

/// Build a legacy single-file project with real media, gap metadata and a
/// completed detect+fit run (anchors, map, fit summary stored).
fn build_legacy(dir: &Path, name: &str) -> LegacyFixture {
    let h = hits(DURATION);
    let tone = Some((440.0f32, 1.0, DURATION - 1.0));
    let ref_gaps = vec![Gap {
        start: 10.0,
        end: 12.0,
    }];
    let member_gaps = vec![Gap {
        start: 20.0,
        end: 21.0,
    }];
    let reference = render_reference(&h, tone, &ref_gaps, DURATION, RATE, 2);
    let model = ClockModel {
        delay_secs: DELAY,
        rate_ratio: 1.0 + DRIFT_PPM * 1e-6,
        nominal_rate: RATE,
    };
    let member = render_member(&h, tone, &model, &member_gaps, DURATION, 2);
    let ref_path = dir.join("reference.wav");
    let mem_path = dir.join("member.wav");
    write_wav_f32(&ref_path, &reference).unwrap();
    write_wav_f32(&mem_path, &member).unwrap();

    let mut project = Project::new(name);
    project.add_track(align_core::project::TrackMeta {
        id: "track-ref".into(),
        name: "reference".into(),
        path: PathBuf::from("reference.wav"),
        nominal_rate: RATE,
        channels: 2,
        frames: reference.frames(),
        source_hash: sha256_file(&ref_path).unwrap(),
    });
    project.add_track(align_core::project::TrackMeta {
        id: "track-member-0".into(),
        name: "member".into(),
        path: PathBuf::from("member.wav"),
        nominal_rate: RATE,
        channels: 2,
        frames: member.frames(),
        source_hash: sha256_file(&mem_path).unwrap(),
    });
    project.reference_gaps = vec![(10.0, 12.0)];
    project
        .member_gaps
        .insert("track-member-0".into(), vec![(20.0, 21.0)]);

    let mut anchors = detect_anchors(
        "track-member-0",
        &reference,
        &member,
        &ref_gaps,
        &member_gaps,
        &PipelineConfig::default(),
    )
    .unwrap();
    let (map, fit) = fit_track("track-member-0", &mut anchors, member.duration_secs()).unwrap();
    let unknown_ranges: Vec<(f64, f64)> = anchors
        .iter()
        .filter(|a| !a.is_usable())
        .map(|a| (a.member_time - 0.1, a.member_time + 0.1))
        .collect();
    project.fit_summaries.push(FitSummary {
        track_id: "track-member-0".into(),
        delay_secs: fit.delay_secs,
        drift_ppm: fit.drift_ppm,
        rms_residual_secs: fit.rms_residual_secs,
        max_abs_residual_secs: fit.max_abs_residual_secs,
        anchors_used: fit.used,
        anchors_rejected: fit.rejected,
        anchors_unknown: anchors.iter().filter(|a| !a.is_usable()).count(),
        known_fraction: map.known_fraction(member.duration_secs(), &unknown_ranges),
    });
    project.anchors.extend(anchors);
    project.maps.push(map);

    let project_path = dir.join("project.json");
    project.save(&project_path).unwrap();
    LegacyFixture {
        project_path,
        project,
        member,
    }
}

fn read_manifest(pkg: &Path) -> PackageManifest {
    let text = std::fs::read_to_string(pkg.join(MANIFEST_FILE)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn opts(mode: MediaMode, force: bool) -> MigrationOptions {
    MigrationOptions {
        mode,
        force,
        probe_hz: Some(440.0),
    }
}

#[test]
fn copy_mode_migration_preserves_semantics() {
    let dir = temp_dir("copy");
    let fix = build_legacy(&dir, "copy-case");
    let pkg = dir.join("copy-case.bandalign");

    let report = migrate_package(&fix.project_path, &pkg, &opts(MediaMode::Copy, false)).unwrap();
    assert_eq!(report.outcome, MigrationOutcome::Completed);
    assert_eq!(report.from_format_version, 1);
    assert_eq!(report.to_format_version, PACKAGE_FORMAT_VERSION);
    assert!(report.discrepancies.is_empty());
    assert!(report.error.is_none());

    // Manifest and packaged project carry the versioned format.
    let manifest = read_manifest(&pkg);
    assert_eq!(manifest.format_version, PACKAGE_FORMAT_VERSION);
    assert_eq!(manifest.media_mode, MediaMode::Copy);
    assert_eq!(manifest.tracks.len(), 2);
    let packaged = Project::load(&pkg.join(PROJECT_FILE)).unwrap();
    assert_eq!(packaged.version, PACKAGE_PROJECT_VERSION);

    // Full semantics survive: reference track, source digests, anchor
    // confidence lifecycle, continuous time map, gaps, epoch.
    assert_eq!(packaged.epoch, fix.project.epoch);
    assert_eq!(
        packaged.reference_track_id, fix.project.reference_track_id
    );
    assert!(compare_projects(&fix.project, &packaged).is_empty());
    let anchors_json = |p: &Project| serde_json::to_string(&p.anchors).unwrap();
    assert_eq!(anchors_json(&packaged), anchors_json(&fix.project));
    let maps_json = |p: &Project| serde_json::to_string(&p.maps).unwrap();
    assert_eq!(maps_json(&packaged), maps_json(&fix.project));

    // Media was copied and matches the recorded source hashes.
    for t in &manifest.tracks {
        let copied = pkg.join(&t.packaged_path);
        assert!(copied.starts_with(pkg.join(MEDIA_DIR)));
        assert_eq!(sha256_file(&copied).unwrap(), t.source_hash);
    }
    align_core::package::validate_package_media(&pkg).unwrap();

    // The alignment actually ran at the target path; residuals and channel
    // phase are recorded as execution evidence.
    let evidence = report.evidence.expect("evidence recorded");
    assert_eq!(evidence.tracks.len(), 1);
    let tev = &evidence.tracks[0];
    assert!(tev.rms_residual_secs <= 0.002, "rms {:?}", tev);
    assert!(tev.channel_max_diff <= 1e-6, "channel {:?}", tev);
    assert!(pkg.join(EVIDENCE_DIR).join("aligned").is_dir());
    assert!(pkg.join(EVIDENCE_DIR).join("evidence.json").is_file());
    let preservation = report.preservation.expect("preservation recorded");
    assert_eq!(preservation.verdict, PreservationVerdict::Preserved);
    assert!(preservation.checks.iter().all(|c| c.pass));
    assert!(pkg.join(MIGRATION_FILE).is_file());

    // The storage adapter resolves the package for the normal commands.
    let storage = ProjectStorage::detect(&pkg).unwrap();
    let loaded = storage.load().unwrap();
    assert_eq!(loaded.media_base, pkg);
    assert_eq!(loaded.project.version, PACKAGE_PROJECT_VERSION);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn reference_mode_keeps_media_external() {
    let dir = temp_dir("ref-mode");
    let fix = build_legacy(&dir, "ref-case");
    let pkg = dir.join("out").join("ref-case.bandalign");
    std::fs::create_dir_all(pkg.parent().unwrap()).unwrap();

    let report =
        migrate_package(&fix.project_path, &pkg, &opts(MediaMode::Reference, false)).unwrap();
    assert_eq!(report.outcome, MigrationOutcome::Completed);

    // No media is copied; tracks are absolute references with valid hashes.
    assert!(!pkg.join(MEDIA_DIR).exists());
    let manifest = read_manifest(&pkg);
    assert_eq!(manifest.media_mode, MediaMode::Reference);
    for t in &manifest.tracks {
        assert!(t.packaged_path.is_absolute());
        assert_eq!(sha256_file(&t.packaged_path).unwrap(), t.source_hash);
    }
    align_core::package::validate_package_media(&pkg).unwrap();
    let preservation = report.preservation.unwrap();
    assert_eq!(preservation.verdict, PreservationVerdict::Preserved);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn duplicate_migration_is_a_detected_noop() {
    let dir = temp_dir("dup");
    let fix = build_legacy(&dir, "dup-case");
    let pkg = dir.join("dup-case.bandalign");

    migrate_package(&fix.project_path, &pkg, &opts(MediaMode::Copy, false)).unwrap();
    let project_bytes = std::fs::read(pkg.join(PROJECT_FILE)).unwrap();
    let report_bytes = std::fs::read(pkg.join(MIGRATION_FILE)).unwrap();

    let second = migrate_package(&fix.project_path, &pkg, &opts(MediaMode::Copy, false)).unwrap();
    assert_eq!(second.outcome, MigrationOutcome::AlreadyMigrated);
    assert_eq!(std::fs::read(pkg.join(PROJECT_FILE)).unwrap(), project_bytes);
    assert_eq!(
        std::fs::read(pkg.join(MIGRATION_FILE)).unwrap(),
        report_bytes
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn conflicting_target_requires_force_and_cleans_backup() {
    let dir = temp_dir("force");
    let fix_a = build_legacy(&dir, "case-a");
    let pkg = dir.join("shared.bandalign");
    migrate_package(&fix_a.project_path, &pkg, &opts(MediaMode::Copy, false)).unwrap();

    // A different project migrating to the same target is a conflict.
    let dir_b = temp_dir("force-b");
    let fix_b = build_legacy(&dir_b, "case-b");
    let err = migrate_package(&fix_b.project_path, &pkg, &opts(MediaMode::Copy, false))
        .unwrap_err();
    assert!(matches!(err, AlignError::PackageExists(_)));

    let report =
        migrate_package(&fix_b.project_path, &pkg, &opts(MediaMode::Copy, true)).unwrap();
    assert_eq!(report.outcome, MigrationOutcome::Completed);
    // The replaced package's backup was discarded after validation passed.
    assert!(report.backup_path.is_none());
    assert!(!dir.join("shared.bandalign.backup-1").exists());
    let digest_b = align_core::hash::sha256_hex(&std::fs::read(&fix_b.project_path).unwrap());
    assert_eq!(read_manifest(&pkg).source_digest, digest_b);

    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&dir_b).ok();
}

#[test]
fn failed_migration_rolls_back_and_reports() {
    let dir = temp_dir("rollback");
    let fix = build_legacy(&dir, "broken-case");
    // Break the source: the member media vanishes before migration.
    std::fs::remove_file(dir.join("member.wav")).unwrap();
    let pkg = dir.join("broken.bandalign");

    let err = migrate_package(&fix.project_path, &pkg, &opts(MediaMode::Copy, false))
        .unwrap_err();
    assert!(matches!(err, AlignError::Io(_)));

    // Nothing is left behind but the rollback report.
    assert!(!pkg.exists());
    assert!(!dir.join("broken.bandalign.staging").exists());
    let rollback_file = dir.join("broken.bandalign.rollback-report.json");
    assert!(rollback_file.is_file());
    let text = std::fs::read_to_string(rollback_file).unwrap();
    let report: align_core::package::MigrationReport =
        serde_json::from_str(&text).unwrap();
    assert_eq!(report.outcome, MigrationOutcome::RolledBack);
    assert!(report.error.is_some());
    let names: Vec<&str> = report.steps.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"rollback-staging"), "steps: {names:?}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn failed_force_replace_restores_existing_package() {
    let dir = temp_dir("restore");
    let fix_a = build_legacy(&dir, "good-case");
    let pkg = dir.join("shared.bandalign");
    migrate_package(&fix_a.project_path, &pkg, &opts(MediaMode::Copy, false)).unwrap();
    let digest_a = read_manifest(&pkg).source_digest.clone();
    let project_bytes = std::fs::read(pkg.join(PROJECT_FILE)).unwrap();

    let dir_b = temp_dir("restore-b");
    let fix_b = build_legacy(&dir_b, "broken-b");
    std::fs::remove_file(dir_b.join("member.wav")).unwrap();
    let err = migrate_package(&fix_b.project_path, &pkg, &opts(MediaMode::Copy, true))
        .unwrap_err();
    assert!(matches!(err, AlignError::Io(_)));

    // The previous package is back, untouched.
    assert_eq!(read_manifest(&pkg).source_digest, digest_a);
    assert_eq!(std::fs::read(pkg.join(PROJECT_FILE)).unwrap(), project_bytes);
    assert!(!dir.join("shared.bandalign.backup-1").exists());

    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&dir_b).ok();
}

#[test]
fn invalid_anchors_and_gaps_are_carried_not_repaired() {
    let dir = temp_dir("nofix");
    let mut fix = build_legacy(&dir, "messy-case");
    // Corrupt the project the way real old projects are: unknown and
    // rejected anchors, a manual override, empty candidate lists.
    fix.project.anchors.push(Anchor {
        id: "a-unknown".into(),
        track_id: "track-member-0".into(),
        ref_time: 11.0,
        member_time: 11.12,
        lag_secs: 0.12,
        status: AnchorStatus::Unknown,
        candidates: vec![],
        confidence: 0.0,
        residual_secs: None,
        note: Some("inside reference gap".into()),
    });
    fix.project.anchors.push(Anchor {
        id: "a-rejected".into(),
        track_id: "track-member-0".into(),
        ref_time: 6.0,
        member_time: 6.35,
        lag_secs: 0.35,
        status: AnchorStatus::Rejected,
        candidates: vec![],
        confidence: 0.2,
        residual_secs: Some(0.23),
        note: None,
    });
    fix.project.anchors[0].status = AnchorStatus::Manual;
    fix.project.anchors[0].residual_secs = None;
    fix.project.save(&fix.project_path).unwrap();
    let before = serde_json::to_string(&fix.project.anchors).unwrap();

    let pkg = dir.join("messy.bandalign");
    let report = migrate_storage(&fix.project_path, &pkg, MediaMode::Copy, false).unwrap();
    assert_eq!(report.outcome, MigrationOutcome::Completed);
    let packaged = Project::load(&pkg.join(PROJECT_FILE)).unwrap();

    // Byte-identical anchors and gaps: nothing normalised, dropped or fixed.
    assert_eq!(serde_json::to_string(&packaged.anchors).unwrap(), before);
    assert_eq!(packaged.reference_gaps, fix.project.reference_gaps);
    assert_eq!(packaged.member_gaps, fix.project.member_gaps);
    let unknown = packaged
        .anchors
        .iter()
        .find(|a| a.id == "a-unknown")
        .unwrap();
    assert_eq!(unknown.status, AnchorStatus::Unknown);
    let rejected = packaged
        .anchors
        .iter()
        .find(|a| a.id == "a-rejected")
        .unwrap();
    assert_eq!(rejected.status, AnchorStatus::Rejected);
    assert_eq!(rejected.residual_secs, Some(0.23));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn reference_switch_epoch_survives_migration() {
    let dir = temp_dir("refswitch");
    let mut fix = build_legacy(&dir, "switch-case");
    // Add a second member, then make it the reference: anchors and maps are
    // cleared and the epoch bumps, exactly the state a switched project has.
    let member2 = fix.member.clone();
    let mem2_path = dir.join("member2.wav");
    write_wav_f32(&mem2_path, &member2).unwrap();
    fix.project.add_track(align_core::project::TrackMeta {
        id: "track-member-1".into(),
        name: "member2".into(),
        path: PathBuf::from("member2.wav"),
        nominal_rate: RATE,
        channels: 2,
        frames: member2.frames(),
        source_hash: sha256_file(&mem2_path).unwrap(),
    });
    let epoch_before = fix.project.epoch;
    fix.project.set_reference("track-member-0").unwrap();
    assert!(fix.project.epoch > epoch_before);
    assert!(fix.project.anchors.is_empty());
    fix.project.save(&fix.project_path).unwrap();

    let pkg = dir.join("switched.bandalign");
    migrate_storage(&fix.project_path, &pkg, MediaMode::Reference, false).unwrap();
    let packaged = Project::load(&pkg.join(PROJECT_FILE)).unwrap();
    assert_eq!(packaged.epoch, fix.project.epoch);
    assert_eq!(
        packaged.reference_track_id.as_deref(),
        Some("track-member-0")
    );
    assert!(packaged.anchors.is_empty());
    assert!(packaged.maps.is_empty());
    assert!(compare_projects(&fix.project, &packaged).is_empty());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn evidence_failure_is_not_masked_by_defaults() {
    let dir = temp_dir("evidence-fail");
    let fix = build_legacy(&dir, "evidence-case");
    let pkg = dir.join("evidence.bandalign");
    migrate_package(&fix.project_path, &pkg, &opts(MediaMode::Copy, false)).unwrap();

    // Destroy the packaged media; the evidence run must fail loudly instead
    // of emitting NaN/zero metrics.
    let manifest = read_manifest(&pkg);
    let member_entry = manifest
        .tracks
        .iter()
        .find(|t| t.track_id == "track-member-0")
        .unwrap();
    std::fs::remove_file(pkg.join(&member_entry.packaged_path)).unwrap();
    let storage = ProjectStorage::detect(&pkg).unwrap();
    let err = run_alignment_evidence(&storage, &dir.join("ev-out"), Some(440.0))
        .unwrap_err();
    assert!(matches!(err, AlignError::Wav(_) | AlignError::Io(_)));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn preservation_degrades_when_residuals_regress() {
    let dir = temp_dir("degraded");
    let fix = build_legacy(&dir, "degraded-case");
    let pkg = dir.join("degraded.bandalign");
    let report =
        migrate_package(&fix.project_path, &pkg, &opts(MediaMode::Copy, false)).unwrap();
    let evidence = report.evidence.unwrap();

    // A stored summary that no longer matches the fresh evidence must flip
    // the verdict to Degraded; the judge never papers over the mismatch.
    let mut old = fix.project.clone();
    old.fit_summaries[0].delay_secs += 0.5;
    let preservation = judge_preservation(&old, &evidence);
    assert_eq!(preservation.verdict, PreservationVerdict::Degraded);
    assert!(preservation.checks.iter().any(|c| !c.pass));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn tampered_package_is_caught_by_static_comparison() {
    let dir = temp_dir("tamper");
    let fix = build_legacy(&dir, "tamper-case");
    let pkg = dir.join("tamper.bandalign");
    migrate_storage(&fix.project_path, &pkg, MediaMode::Copy, false).unwrap();

    let mut packaged = Project::load(&pkg.join(PROJECT_FILE)).unwrap();
    packaged.anchors[0].confidence = 0.01;
    packaged.epoch += 1;
    let d = compare_projects(&fix.project, &packaged);
    assert!(d.iter().any(|m| m.contains("anchors")), "{d:?}");
    assert!(d.iter().any(|m| m.contains("epoch")), "{d:?}");

    std::fs::remove_dir_all(&dir).ok();
}
