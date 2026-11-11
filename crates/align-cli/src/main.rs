use align_core::audio::{write_wav_f32, AudioBuffer, AudioReader};
use align_core::estimate::fit_track;
use align_core::pipeline::{detect_anchors, PipelineConfig};
use align_core::render::{mix_stems, render_project, RenderOptions};
use align_core::synth::{
    render_member, render_reference, ClockModel, Gap, Hit, HitKind,
};
use align_core::verify::{
    channel_consistency, estimator_error, phase_continuity, GroundTruth,
};
use align_core::Project;
use std::path::PathBuf;

fn main() {
    if let Err(e) = run() {
        eprintln!("align-cli error: {e}");
        std::process::exit(1);
    }
}

fn run() -> align_core::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("synthesize") => cmd_synthesize(&args[2..]),
        Some("init-project") => cmd_init_project(&args[2..]),
        Some("align") => cmd_align(&args[2..]),
        Some("export-positions") => cmd_export_positions(&args[2..]),
        Some("verify-project") => cmd_verify_project(&args[2..]),
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage:\n\
         align-cli synthesize --out DIR [--rate 48000] [--duration 60] [--delay 0.12] \
         [--drift-ppm 75] [--channels 2] [--ref-gap 20:24] [--member-gap 35:37]\n\
         align-cli init-project --name N --reference WAV --member WAV [--member WAV ...] --out FILE\n\
         align-cli align --project FILE --out DIR\n\
         align-cli export-positions --project FILE --out positions.csv\n\
         align-cli verify-project --project FILE --truth FILE --out FILE"
    );
}

/// Audit trail: for every anchor, the unadjusted member position, the fitted
/// project position, the lag and residual, plus status. The original media is
/// never modified, so the pre-alignment timing is always recoverable.
fn cmd_export_positions(args: &[String]) -> align_core::Result<()> {
    let mut project_path = None;
    let mut out_path = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--project" => project_path = Some(PathBuf::from(&args[i + 1])),
            "--out" => out_path = Some(PathBuf::from(&args[i + 1])),
            other => panic!("unknown export-positions flag: {other}"),
        }
        i += 2;
    }
    let project = Project::load(&project_path.unwrap())?;
    let mut csv = String::from(
        "anchor_id,track_id,ref_time_secs,member_original_secs,lag_secs,mapped_project_secs,residual_secs,status,confidence\n",
    );
    for a in &project.anchors {
        let mapped = project
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
                .unwrap_or_else(|| "".into()),
            a.status,
            a.confidence
        ));
    }
    if let Some(p) = out_path {
        std::fs::write(p, csv)?;
    } else {
        print!("{csv}");
    }
    Ok(())
}

fn cmd_init_project(args: &[String]) -> align_core::Result<()> {
    let mut name = "band-rehearsal".to_string();
    let mut reference: Option<PathBuf> = None;
    let mut members: Vec<PathBuf> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut ref_gaps: Vec<(f64, f64)> = Vec::new();
    let mut member_gaps: Vec<(f64, f64)> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => name = args[i + 1].clone(),
            "--reference" => reference = Some(PathBuf::from(&args[i + 1])),
            "--member" => members.push(PathBuf::from(&args[i + 1])),
            "--out" => out = Some(PathBuf::from(&args[i + 1])),
            "--ref-gap" => ref_gaps.push(parse_gap_pair(&args[i + 1])),
            "--member-gap" => member_gaps.push(parse_gap_pair(&args[i + 1])),
            other => panic!("unknown init-project flag: {other}"),
        }
        i += 2;
    }
    let reference = reference.expect("--reference required");
    let out = out.expect("--out required");

    let mut project = Project::new(name);
    let project_dir = out.parent().unwrap_or_else(|| std::path::Path::new("."));
    let mut add = |path: &PathBuf, id: &str| -> align_core::Result<()> {
        let buf = AudioReader::load(path)?;
        let rel = path
            .strip_prefix(project_dir)
            .map(PathBuf::from)
            .unwrap_or_else(|_| path.clone());
        let source_hash = align_core::hash::sha256_file(path)?;
        project.add_track(align_core::project::TrackMeta {
            id: id.to_string(),
            name: path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| id.to_string()),
            path: rel,
            nominal_rate: buf.sample_rate,
            channels: buf.channels,
            frames: buf.frames(),
            source_hash,
        });
        Ok(())
    };
    add(&reference, "track-ref")?;
    for (k, m) in members.iter().enumerate() {
        add(m, &format!("track-member-{k}"))?;
    }
    project.reference_gaps = ref_gaps;
    let first_member_id = project
        .member_tracks()
        .next()
        .map(|t| t.id.clone());
    if let Some(id) = first_member_id {
        project.member_gaps.insert(id, member_gaps.clone());
    }
    project.save(&out)?;
    println!("wrote project {}", out.display());
    Ok(())
}

fn parse_gap_pair(spec: &str) -> (f64, f64) {
    let g = parse_gap(spec);
    (g.start, g.end)
}

struct SynthArgs {
    out: PathBuf,
    rate: u32,
    duration: f64,
    delay: f64,
    drift_ppm: f64,
    channels: u16,
    ref_gaps: Vec<Gap>,
    member_gaps: Vec<Gap>,
}

fn parse_synth(args: &[String]) -> SynthArgs {
    let mut s = SynthArgs {
        out: PathBuf::from("synth-out"),
        rate: 48_000,
        duration: 60.0,
        delay: 0.12,
        drift_ppm: 75.0,
        channels: 2,
        ref_gaps: Vec::new(),
        member_gaps: Vec::new(),
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => s.out = PathBuf::from(&args[i + 1]),
            "--rate" => s.rate = args[i + 1].parse().unwrap(),
            "--duration" => s.duration = args[i + 1].parse().unwrap(),
            "--delay" => s.delay = args[i + 1].parse().unwrap(),
            "--drift-ppm" => s.drift_ppm = args[i + 1].parse().unwrap(),
            "--channels" => s.channels = args[i + 1].parse().unwrap(),
            "--ref-gap" => s.ref_gaps.push(parse_gap(&args[i + 1])),
            "--member-gap" => s.member_gaps.push(parse_gap(&args[i + 1])),
            other => panic!("unknown synth flag: {other}"),
        }
        i += 2;
    }
    s
}

fn parse_gap(spec: &str) -> Gap {
    let (a, b) = spec.split_once(':').expect("gap must be start:end");
    Gap {
        start: a.parse().unwrap(),
        end: b.parse().unwrap(),
    }
}

fn build_hits(duration: f64) -> Vec<Hit> {
    // Claps every 4 s across the whole timeline give anchors spanning the
    // full recording; interspersed drums exercise multiple correlation peaks.
    let mut hits = Vec::new();
    let mut t = 2.0;
    let mut k = 0;
    while t < duration - 1.0 {
        hits.push(Hit {
            time: t,
            kind: HitKind::Clap,
        });
        if k % 3 == 1 {
            hits.push(Hit {
                time: t + 0.5,
                kind: HitKind::Drum(180.0 + 20.0 * (k % 4) as f32),
            });
        }
        t += 4.0;
        k += 1;
    }
    hits
}

fn cmd_synthesize(args: &[String]) -> align_core::Result<()> {
    let s = parse_synth(args);
    std::fs::create_dir_all(&s.out)?;

    let hits = build_hits(s.duration);
    let tone = Some((440.0f32, 1.0, s.duration - 1.0));
    let reference = render_reference(
        &hits,
        tone,
        &s.ref_gaps,
        s.duration,
        s.rate,
        s.channels,
    );
    let model = ClockModel {
        delay_secs: s.delay,
        rate_ratio: 1.0 + s.drift_ppm * 1e-6,
        nominal_rate: s.rate,
    };
    let member = render_member(
        &hits,
        tone,
        &model,
        &s.member_gaps,
        s.duration,
        s.channels,
    );

    let ref_path = s.out.join("reference.wav");
    let mem_path = s.out.join("member.wav");
    write_wav_f32(&ref_path, &reference)?;
    write_wav_f32(&mem_path, &member)?;

    let truth = GroundTruth {
        delay_secs: s.delay,
        rate_ratio: model.rate_ratio,
    };
    std::fs::write(
        s.out.join("truth.json"),
        serde_json::to_string_pretty(&truth)?,
    )?;
    println!(
        "wrote {} ({}s), {} ({}s), truth.json",
        ref_path.display(),
        reference.duration_secs(),
        mem_path.display(),
        member.duration_secs()
    );
    Ok(())
}

fn cmd_align(args: &[String]) -> align_core::Result<()> {
    let mut project_path = None;
    let mut out = PathBuf::from("aligned-out");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--project" => project_path = Some(PathBuf::from(&args[i + 1])),
            "--out" => out = PathBuf::from(&args[i + 1]),
            other => panic!("unknown align flag: {other}"),
        }
        i += 2;
    }
    let project_path = project_path.expect("--project required");
    let mut project = Project::load(&project_path)?;
    std::fs::create_dir_all(&out)?;

    let base = project_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let ref_meta = project.reference()?.clone();
    let reference = AudioReader::load(&base.join(&ref_meta.path))?;

    let config = PipelineConfig::default();
    let render_opts = RenderOptions {
        output_rate: project.settings.project_rate,
        kernel_taps_half: project.settings.kernel_taps_half,
        ..RenderOptions::default()
    };
    let ref_gaps: Vec<Gap> = project
        .reference_gaps
        .iter()
        .map(|(s, e)| Gap { start: *s, end: *e })
        .collect();

    let mut sources: Vec<(AudioBuffer, align_core::TimeMap)> = Vec::new();
    let member_ids: Vec<String> = project.member_tracks().map(|t| t.id.clone()).collect();
    for track_id in &member_ids {
        let meta = project
            .tracks
            .iter()
            .find(|t| &t.id == track_id)
            .unwrap()
            .clone();
        let member = AudioReader::load(&base.join(&meta.path))?;
        let member_gaps: Vec<Gap> = project
            .member_gaps
            .get(track_id)
            .map(|gs| gs.iter().map(|(s, e)| Gap { start: *s, end: *e }).collect())
            .unwrap_or_default();
        let mut anchors = detect_anchors(
            track_id,
            &reference,
            &member,
            &ref_gaps,
            &member_gaps,
            &config,
        )?;

        let duration = member.duration_secs();
        let (map, fit) = if project.settings.correct_drift {
            fit_track(track_id, &mut anchors, duration)?
        } else {
            let m = align_core::TimeMap::from_constant_delay(
                track_id,
                anchors
                    .iter()
                    .find(|a| a.is_usable())
                    .map(|a| a.lag_secs)
                    .unwrap_or(0.0),
                duration,
            );
            let fit = align_core::estimate::FitResult {
                slope: 1.0,
                offset: 0.0,
                delay_secs: anchors
                    .iter()
                    .find(|a| a.is_usable())
                    .map(|a| a.lag_secs)
                    .unwrap_or(0.0),
                drift_ppm: 0.0,
                rms_residual_secs: 0.0,
                max_abs_residual_secs: 0.0,
                used: anchors.iter().filter(|a| a.is_usable()).count(),
                rejected: 0,
            };
            (m, fit)
        };

        let unknown = anchors.iter().filter(|a| !a.is_usable()).count();
        let unknown_ranges: Vec<(f64, f64)> = anchors
            .iter()
            .filter(|a| !a.is_usable())
            .map(|a| (a.member_time - 0.1, a.member_time + 0.1))
            .collect();
        project.fit_summaries.push(align_core::project::FitSummary {
            track_id: track_id.clone(),
            delay_secs: fit.delay_secs,
            drift_ppm: fit.drift_ppm,
            rms_residual_secs: fit.rms_residual_secs,
            max_abs_residual_secs: fit.max_abs_residual_secs,
            anchors_used: fit.used,
            anchors_rejected: fit.rejected,
            anchors_unknown: unknown,
            known_fraction: map.known_fraction(duration, &unknown_ranges),
        });
        project.anchors.extend(anchors);
        project.maps.push(map.clone());
        sources.push((member, map));
    }

    let refs_map = align_core::TimeMap::from_constant_delay(
        &ref_meta.id,
        0.0,
        reference.duration_secs(),
    );
    let all_sources: Vec<(&AudioBuffer, &align_core::TimeMap)> =
        std::iter::once((&reference, &refs_map))
            .chain(sources.iter().map(|(b, m)| (b, m)))
            .collect();
    let (rendered, project_end) = render_project(&all_sources, &render_opts)?;

    let mut reports = Vec::new();
    for (i, rt) in rendered.iter().enumerate() {
        let name = if i == 0 {
            "reference_aligned.wav".to_string()
        } else {
            format!("{}_aligned.wav", member_ids[i - 1])
        };
        write_wav_f32(&out.join(&name), &rt.buffer)?;
        reports.push(rt.report.clone());
    }
    let mix_inputs: Vec<&AudioBuffer> = rendered.iter().map(|r| &r.buffer).collect();
    let mix = mix_stems(&mix_inputs);
    write_wav_f32(&out.join("mix_preview.wav"), &mix)?;

    // Before/after preview: raw member then aligned member, same project rate.
    if let Some((buf, _)) = sources.first() {
        let mut ab = AudioBuffer::new(render_opts.output_rate, buf.channels);
        ab.data = buf.data.clone();
        let gap = (2.0 * render_opts.output_rate as f64) as usize * ab.channels as usize;
        let mut before_after = ab;
        before_after
            .data
            .extend(std::iter::repeat_n(0.0f32, gap));
        if let Some(aligned) = rendered.get(1) {
            before_after.data.extend(aligned.buffer.data.iter().copied());
        }
        write_wav_f32(&out.join("before_after.wav"), &before_after)?;
    }

    project.save(&project_path)?;
    std::fs::write(
        out.join("render_reports.json"),
        serde_json::to_string_pretty(&reports)?,
    )?;
    println!(
        "aligned {} track(s); project end {:.3}s; reports in {}",
        rendered.len(),
        project_end,
        out.display()
    );
    Ok(())
}

fn cmd_verify_project(args: &[String]) -> align_core::Result<()> {
    let mut project_path = None;
    let mut truth_path = None;
    let mut out_path = None;
    let mut aligned_dir = PathBuf::from("aligned-out");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--project" => project_path = Some(PathBuf::from(&args[i + 1])),
            "--truth" => truth_path = Some(PathBuf::from(&args[i + 1])),
            "--out" => out_path = Some(PathBuf::from(&args[i + 1])),
            "--aligned-dir" => aligned_dir = PathBuf::from(&args[i + 1]),
            other => panic!("unknown verify flag: {other}"),
        }
        i += 2;
    }
    let project = Project::load(&project_path.unwrap())?;
    let truth: GroundTruth =
        serde_json::from_str(&std::fs::read_to_string(truth_path.unwrap())?)?;
    let map = project
        .maps
        .first()
        .ok_or_else(|| align_core::AlignError::Other("project has no fitted map".into()))?;
    let duration = project
        .tracks
        .iter()
        .find(|t| t.id == map.track_id)
        .map(|t| t.frames as f64 / t.nominal_rate as f64)
        .unwrap_or(60.0);
    let est = estimator_error(map, &truth, duration);

    let joins: Vec<f64> = map
        .segments
        .windows(2)
        .map(|w| w[1].t0)
        .collect();
    let aligned_path = aligned_dir.join(format!("{}_aligned.wav", map.track_id));
    let phase = if aligned_path.exists() {
        let rendered = AudioReader::load(&aligned_path)?;
        phase_continuity(&rendered, 440.0 / truth.rate_ratio, &joins)
    } else {
        align_core::verify::PhaseContinuityReport {
            max_phase_jump_rad: f64::NAN,
            worst_join_secs: 0.0,
        }
    };
    let channel = if aligned_path.exists() {
        channel_consistency(&AudioReader::load(&aligned_path)?)
    } else {
        align_core::verify::ChannelConsistencyReport {
            max_interchannel_diff: f64::NAN,
            rms_interchannel_diff: f64::NAN,
        }
    };

    #[derive(serde::Serialize)]
    struct Verification {
        estimator: align_core::verify::EstimatorError,
        phase: align_core::verify::PhaseContinuityReport,
        channel: align_core::verify::ChannelConsistencyReport,
        summaries: Vec<align_core::project::FitSummary>,
    }
    let v = Verification {
        estimator: est,
        phase,
        channel,
        summaries: project.fit_summaries,
    };
    let json = serde_json::to_string_pretty(&v)?;
    if let Some(p) = out_path {
        std::fs::write(p, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}
