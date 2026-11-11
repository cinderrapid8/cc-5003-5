use align_core::anchor::{Anchor, AnchorStatus};
use align_core::audio::{write_wav_f32, AudioBuffer, AudioReader};
use align_core::estimate::fit_track;
use align_core::pipeline::{detect_anchors, PipelineConfig};
use align_core::project::{Project, TrackMeta};
use align_core::render::{mix_stems, render_project, RenderOptions};
use align_core::synth::Gap;
use align_core::AlignError;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Mutex;

/// Shared application state. The project plus the loaded PCM buffers live here
/// so repeated commands do not re-decode WAV files.
pub struct AppState {
    pub inner: Mutex<State>,
}

pub struct State {
    pub project: Project,
    pub project_dir: PathBuf,
    pub buffers: Vec<NamedBuffer>,
}

pub struct NamedBuffer {
    pub track_id: String,
    pub buffer: AudioBuffer,
}

#[derive(Clone, Debug, Serialize)]
pub struct AnchorView {
    pub id: String,
    pub track_id: String,
    pub ref_time: f64,
    pub member_time: f64,
    pub lag_secs: f64,
    pub status: String,
    pub confidence: f64,
    pub residual_secs: Option<f64>,
    pub candidates: Vec<align_core::anchor::LagCandidate>,
    pub note: Option<String>,
}

impl From<&Anchor> for AnchorView {
    fn from(a: &Anchor) -> Self {
        AnchorView {
            id: a.id.clone(),
            track_id: a.track_id.clone(),
            ref_time: a.ref_time,
            member_time: a.member_time,
            lag_secs: a.lag_secs,
            status: format!("{:?}", a.status),
            confidence: a.confidence,
            residual_secs: a.residual_secs,
            candidates: a.candidates.clone(),
            note: a.note.clone(),
        }
    }
}

fn state_err<T>(e: T) -> String
where
    T: std::fmt::Display,
{
    format!("{e}")
}

#[tauri::command]
pub fn new_project(name: String, state: tauri::State<AppState>) -> Result<(), String> {
    let mut s = state.inner.lock().map_err(state_err)?;
    s.project = Project::new(name);
    s.buffers.clear();
    Ok(())
}

#[tauri::command]
pub fn import_track(
    path: String,
    state: tauri::State<AppState>,
) -> Result<TrackMeta, String> {
    let p = PathBuf::from(&path);
    let buffer = AudioReader::load(&p).map_err(state_err)?;
    let mut s = state.inner.lock().map_err(state_err)?;
    let id = format!("track-{}", s.project.tracks.len());
    let name = p
        .file_stem()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| id.clone());
    let rel = p
        .strip_prefix(&s.project_dir)
        .map(PathBuf::from)
        .unwrap_or_else(|_| p.clone());
    let source_hash = align_core::hash::sha256_file(&p).map_err(state_err)?;
    let meta = TrackMeta {
        id: id.clone(),
        name,
        path: rel,
        nominal_rate: buffer.sample_rate,
        channels: buffer.channels,
        frames: buffer.frames(),
        source_hash,
    };
    s.project.add_track(meta.clone());
    s.buffers.push(NamedBuffer {
        track_id: id,
        buffer,
    });
    Ok(meta)
}

#[tauri::command]
pub fn set_reference(
    track_id: String,
    state: tauri::State<AppState>,
) -> Result<u64, String> {
    let mut s = state.inner.lock().map_err(state_err)?;
    s.project.set_reference(&track_id).map_err(state_err)?;
    Ok(s.project.epoch)
}

#[tauri::command]
pub fn run_detection(state: tauri::State<AppState>) -> Result<Vec<AnchorView>, String> {
    let mut s = state.inner.lock().map_err(state_err)?;
    let ref_id = s
        .project
        .reference_track_id
        .clone()
        .ok_or_else(|| AlignError::NoReference.to_string())?;
    let reference = s
        .buffers
        .iter()
        .find(|b| b.track_id == ref_id)
        .map(|b| &b.buffer)
        .ok_or_else(|| "reference buffer not loaded".to_string())?
        .clone();
    let config = PipelineConfig::default();
    let ref_gaps: Vec<Gap> = s
        .project
        .reference_gaps
        .iter()
        .map(|(a, b)| Gap { start: *a, end: *b })
        .collect();
    let member_ids: Vec<String> = s.project.member_tracks().map(|t| t.id.clone()).collect();
    s.project.anchors.clear();
    for track_id in member_ids {
        let member = s
            .buffers
            .iter()
            .find(|b| b.track_id == track_id)
            .map(|b| &b.buffer)
            .ok_or_else(|| "member buffer not loaded".to_string())?
            .clone();
        let member_gaps: Vec<Gap> = s
            .project
            .member_gaps
            .get(&track_id)
            .map(|gs| gs.iter().map(|(a, b)| Gap { start: *a, end: *b }).collect())
            .unwrap_or_default();
        let anchors = detect_anchors(
            &track_id,
            &reference,
            &member,
            &ref_gaps,
            &member_gaps,
            &config,
        )
        .map_err(state_err)?;
        s.project.anchors.extend(anchors);
    }
    s.project.bump_epoch();
    Ok(s.project.anchors.iter().map(AnchorView::from).collect())
}

/// User chooses one of several correlation candidates for an anchor.
#[tauri::command]
pub fn pick_candidate(
    anchor_id: String,
    candidate_index: usize,
    state: tauri::State<AppState>,
) -> Result<AnchorView, String> {
    let mut s = state.inner.lock().map_err(state_err)?;
    let a = s
        .project
        .anchors
        .iter_mut()
        .find(|a| a.id == anchor_id)
        .ok_or_else(|| AlignError::AnchorNotFound(anchor_id.clone()).to_string())?;
    if !a.pick_candidate(candidate_index) {
        return Err("candidate index out of range".into());
    }
    Ok(AnchorView::from(a))
}

/// Manual correction: place the member-side time explicitly.
#[tauri::command]
pub fn set_manual_anchor(
    anchor_id: String,
    member_time: f64,
    state: tauri::State<AppState>,
) -> Result<AnchorView, String> {
    let mut s = state.inner.lock().map_err(state_err)?;
    let a = s
        .project
        .anchors
        .iter_mut()
        .find(|a| a.id == anchor_id)
        .ok_or_else(|| AlignError::AnchorNotFound(anchor_id.clone()).to_string())?;
    a.set_manual(member_time);
    Ok(AnchorView::from(a))
}

#[tauri::command]
pub fn mark_anchor_unknown(
    anchor_id: String,
    state: tauri::State<AppState>,
) -> Result<AnchorView, String> {
    let mut s = state.inner.lock().map_err(state_err)?;
    let a = s
        .project
        .anchors
        .iter_mut()
        .find(|a| a.id == anchor_id)
        .ok_or_else(|| AlignError::AnchorNotFound(anchor_id.clone()).to_string())?;
    a.mark_unknown();
    Ok(AnchorView::from(a))
}

#[derive(Clone, Debug, Serialize)]
pub struct FitView {
    pub track_id: String,
    pub delay_secs: f64,
    pub drift_ppm: f64,
    pub rms_residual_secs: f64,
    pub max_abs_residual_secs: f64,
    pub anchors_used: usize,
    pub anchors_rejected: usize,
    pub anchors_unknown: usize,
    pub anchors: Vec<AnchorView>,
}

#[tauri::command]
pub fn fit_all(state: tauri::State<AppState>) -> Result<Vec<FitView>, String> {
    let mut s = state.inner.lock().map_err(state_err)?;
    s.project.maps.clear();
    s.project.fit_summaries.clear();
    let member_ids: Vec<String> = s.project.member_tracks().map(|t| t.id.clone()).collect();
    let mut views = Vec::new();
    for track_id in member_ids {
        let duration = s
            .buffers
            .iter()
            .find(|b| b.track_id == track_id)
            .map(|b| b.buffer.duration_secs())
            .ok_or_else(|| "buffer missing".to_string())?;
        let (map, fit) = {
            // A previous fit may have marked outliers Rejected. Reconsider them
            // by restoring Rejected anchors to Auto before refitting; Unknown
            // anchors stay out because the user or detector distrusts them.
            for a in s
                .project
                .anchors
                .iter_mut()
                .filter(|a| a.track_id == track_id)
            {
                if a.status == AnchorStatus::Rejected {
                    a.status = AnchorStatus::Auto;
                }
                a.residual_secs = None;
            }
            let mut owned: Vec<Anchor> = s
                .project
                .anchors
                .iter()
                .filter(|a| a.track_id == track_id)
                .cloned()
                .collect();
            let (m, f) = fit_track(&track_id, &mut owned, duration).map_err(state_err)?;
            // Write residuals and statuses back into the project anchors.
            for updated in owned {
                if let Some(slot) = s
                    .project
                    .anchors
                    .iter_mut()
                    .find(|a| a.id == updated.id)
                {
                    slot.status = updated.status;
                    slot.residual_secs = updated.residual_secs;
                }
            }
            (m, f)
        };
        let unknown_ranges: Vec<(f64, f64)> = s
            .project
            .anchors
            .iter()
            .filter(|a| a.track_id == track_id && !a.is_usable())
            .map(|a| (a.member_time - 0.1, a.member_time + 0.1))
            .collect();
        let unknown = unknown_ranges.len();
        let known_fraction = map.known_fraction(duration, &unknown_ranges);
        s.project.maps.push(map);
        s.project.fit_summaries
            .push(align_core::project::FitSummary {
                track_id: track_id.clone(),
                delay_secs: fit.delay_secs,
                drift_ppm: fit.drift_ppm,
                rms_residual_secs: fit.rms_residual_secs,
                max_abs_residual_secs: fit.max_abs_residual_secs,
                anchors_used: fit.used,
                anchors_rejected: fit.rejected,
                anchors_unknown: unknown,
                known_fraction,
            });
        views.push(FitView {
            track_id: track_id.clone(),
            delay_secs: fit.delay_secs,
            drift_ppm: fit.drift_ppm,
            rms_residual_secs: fit.rms_residual_secs,
            max_abs_residual_secs: fit.max_abs_residual_secs,
            anchors_used: fit.used,
            anchors_rejected: fit.rejected,
            anchors_unknown: unknown,
            anchors: s
                .project
                .anchors
                .iter()
                .filter(|a| a.track_id == track_id)
                .map(AnchorView::from)
                .collect(),
        });
    }
    s.project.bump_epoch();
    Ok(views)
}

#[derive(Clone, Debug, Serialize)]
pub struct RenderView {
    pub output_dir: String,
    pub project_end_secs: f64,
    pub reports: Vec<align_core::render::RenderReport>,
}

#[tauri::command]
pub fn render_aligned(
    output_dir: String,
    state: tauri::State<AppState>,
) -> Result<RenderView, String> {
    let s = state.inner.lock().map_err(state_err)?;
    let dir = PathBuf::from(&output_dir);
    std::fs::create_dir_all(&dir).map_err(state_err)?;
    let opts = RenderOptions {
        output_rate: s.project.settings.project_rate,
        kernel_taps_half: s.project.settings.kernel_taps_half,
        ..RenderOptions::default()
    };
    let ref_id = s
        .project
        .reference_track_id
        .clone()
        .ok_or_else(|| AlignError::NoReference.to_string())?;
    let reference = s
        .buffers
        .iter()
        .find(|b| b.track_id == ref_id)
        .map(|b| &b.buffer)
        .ok_or_else(|| "reference buffer missing".to_string())?;
    let ref_map =
        align_core::TimeMap::from_constant_delay(&ref_id, 0.0, reference.duration_secs());
    let mut pairs: Vec<(&AudioBuffer, &align_core::TimeMap)> = vec![(reference, &ref_map)];
    for map in &s.project.maps {
        let buf = s
            .buffers
            .iter()
            .find(|b| b.track_id == map.track_id)
            .map(|b| &b.buffer)
            .ok_or_else(|| "member buffer missing".to_string())?;
        pairs.push((buf, map));
    }
    let (rendered, project_end) = render_project(&pairs, &opts).map_err(state_err)?;
    let mut reports = Vec::new();
    let mut stem_paths = Vec::new();
    for (i, rt) in rendered.iter().enumerate() {
        let name = if i == 0 {
            "reference_aligned.wav".to_string()
        } else {
            format!("{}_aligned.wav", rt.report.track_id)
        };
        let path = dir.join(&name);
        write_wav_f32(&path, &rt.buffer).map_err(state_err)?;
        stem_paths.push(path);
        reports.push(rt.report.clone());
    }
    let mix_inputs: Vec<&AudioBuffer> = rendered.iter().map(|r| &r.buffer).collect();
    let mix = mix_stems(&mix_inputs);
    write_wav_f32(&dir.join("mix_preview.wav"), &mix).map_err(state_err)?;
    Ok(RenderView {
        output_dir,
        project_end_secs: project_end,
        reports,
    })
}

#[tauri::command]
pub fn save_project(path: String, state: tauri::State<AppState>) -> Result<(), String> {
    let s = state.inner.lock().map_err(state_err)?;
    s.project
        .save(&PathBuf::from(&path))
        .map_err(state_err)
}

#[tauri::command]
pub fn load_project(path: String, state: tauri::State<AppState>) -> Result<Project, String> {
    let mut s = state.inner.lock().map_err(state_err)?;
    let p = PathBuf::from(&path);
    let project = Project::load(&p).map_err(state_err)?;
    s.project_dir = p.parent().map(PathBuf::from).unwrap_or_default();
    s.buffers.clear();
    for meta in &project.tracks {
        let buffer = AudioReader::load(&s.project_dir.join(&meta.path)).map_err(state_err)?;
        s.buffers.push(NamedBuffer {
            track_id: meta.id.clone(),
            buffer,
        });
    }
    s.project = project.clone();
    Ok(project)
}

#[tauri::command]
pub fn list_anchors(state: tauri::State<AppState>) -> Result<Vec<AnchorView>, String> {
    let s = state.inner.lock().map_err(state_err)?;
    Ok(s.project.anchors.iter().map(AnchorView::from).collect())
}

/// Audit export matching the CLI: original member position, fitted project
/// position, lag, residual and status for every anchor.
#[tauri::command]
pub fn export_positions(path: String, state: tauri::State<AppState>) -> Result<usize, String> {
    let s = state.inner.lock().map_err(state_err)?;
    let mut csv = String::from(
        "anchor_id,track_id,ref_time_secs,member_original_secs,lag_secs,mapped_project_secs,residual_secs,status,confidence\n",
    );
    for a in &s.project.anchors {
        let mapped = s
            .project
            .maps
            .iter()
            .find(|m| m.track_id == a.track_id)
            .and_then(|m| m.to_project(a.member_time).ok());
        csv.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{},{},{:?},{:.4}\n",
            a.id,
            a.track_id,
            a.ref_time,
            a.member_time,
            a.lag_secs,
            mapped
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "unknown".into()),
            a.residual_secs
                .map(|v| format!("{v:.6}"))
                .unwrap_or_default(),
            a.status,
            a.confidence
        ));
    }
    let rows = s.project.anchors.len();
    std::fs::write(std::path::PathBuf::from(&path), csv).map_err(state_err)?;
    Ok(rows)
}
