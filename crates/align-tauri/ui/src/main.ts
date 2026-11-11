import { invoke } from "@tauri-apps/api/core";

export interface LagCandidate {
  lag_secs: number;
  score: number;
}

export interface AnchorView {
  id: string;
  track_id: string;
  ref_time: number;
  member_time: number;
  lag_secs: number;
  status: string;
  confidence: number;
  residual_secs: number | null;
  candidates: LagCandidate[];
  note: string | null;
}

export interface TrackMeta {
  id: string;
  name: string;
  path: string;
  nominal_rate: number;
  channels: number;
  frames: number;
}

export interface FitView {
  track_id: string;
  delay_secs: number;
  drift_ppm: number;
  rms_residual_secs: number;
  max_abs_residual_secs: number;
  anchors_used: number;
  anchors_rejected: number;
  anchors_unknown: number;
  anchors: AnchorView[];
}

export interface RenderReport {
  track_id: string;
  output_frames: number;
  output_duration_secs: number;
  head_pad_frames: number;
  tail_pad_frames: number;
  unknown_frames: number;
  edge_clamped_frames: number;
  min_map_slope: number;
  max_map_slope: number;
}

export interface RenderView {
  output_dir: string;
  project_end_secs: number;
  reports: RenderReport[];
}

let tracks: TrackMeta[] = [];
let anchors: AnchorView[] = [];
let referenceId: string | null = null;

const $ = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

function fmt(n: number, digits = 4): string {
  return n.toFixed(digits);
}

function renderTracks(): void {
  const tbody = $<HTMLTableSectionElement>("tracks-table").querySelector("tbody")!;
  tbody.innerHTML = "";
  for (const t of tracks) {
    const tr = document.createElement("tr");
    const duration = t.frames / t.nominal_rate;
    tr.innerHTML = `
      <td><input type="radio" name="ref" ${t.id === referenceId ? "checked" : ""} /></td>
      <td>${t.name}</td>
      <td>${t.nominal_rate}</td>
      <td>${t.channels}</td>
      <td>${fmt(duration, 2)} s</td>`;
    tr.querySelector("input")!.addEventListener("change", async () => {
      const epoch = await invoke<number>("set_reference", { trackId: t.id });
      referenceId = t.id;
      anchors = [];
      renderAnchors();
      $("epoch-badge").textContent = `epoch ${epoch}`;
      renderTracks();
    });
    tbody.appendChild(tr);
  }
}

function statusClass(status: string): string {
  switch (status) {
    case "Auto":
    case "UserPicked":
    case "Manual":
      return "status-ok";
    case "Rejected":
      return "status-rejected";
    default:
      return "status-unknown";
  }
}

function renderAnchors(): void {
  const tbody = $<HTMLTableSectionElement>("anchors-table").querySelector("tbody")!;
  tbody.innerHTML = "";
  for (const a of anchors) {
    const tr = document.createElement("tr");
    tr.className = statusClass(a.status);
    const residual =
      a.residual_secs === null ? "—" : `${fmt(a.residual_secs * 1000, 3)}`;
    const candidateText = a.candidates
      .map(
        (c, i) =>
          `<button class="cand" data-anchor="${a.id}" data-index="${i}">
             #${i} ${fmt(c.lag_secs * 1000, 2)} ms (${fmt(c.score, 3)})
           </button>`,
      )
      .join("");
    tr.innerHTML = `
      <td>${fmt(a.ref_time, 3)}</td>
      <td>${fmt(a.member_time, 3)}</td>
      <td>${fmt(a.lag_secs * 1000, 2)}</td>
      <td>${fmt(a.confidence, 3)}</td>
      <td>${residual}</td>
      <td>${a.status}${a.note ? `<br/><small>${a.note}</small>` : ""}</td>
      <td>${candidateText || "—"}</td>
      <td>
        <button class="mark-unknown" data-anchor="${a.id}">Mark unknown</button>
      </td>`;
    tbody.appendChild(tr);
  }

  tbody.querySelectorAll<HTMLButtonElement>("button.cand").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const updated = await invoke<AnchorView>("pick_candidate", {
        anchorId: btn.dataset.anchor,
        candidateIndex: Number(btn.dataset.index),
      });
      upsertAnchor(updated);
      renderAnchors();
    });
  });
  tbody
    .querySelectorAll<HTMLButtonElement>("button.mark-unknown")
    .forEach((btn) => {
      btn.addEventListener("click", async () => {
        const updated = await invoke<AnchorView>("mark_anchor_unknown", {
          anchorId: btn.dataset.anchor,
        });
        upsertAnchor(updated);
        renderAnchors();
      });
    });
}

function upsertAnchor(a: AnchorView): void {
  const i = anchors.findIndex((x) => x.id === a.id);
  if (i >= 0) anchors[i] = a;
  else anchors.push(a);
}

function renderFit(views: FitView[]): void {
  const host = $<HTMLDivElement>("fit-summary");
  if (views.length === 0) {
    host.innerHTML = "";
    return;
  }
  host.innerHTML = views
    .map(
      (v) => `
      <div class="fit-card">
        <h3>${v.track_id}</h3>
        <dl>
          <dt>Fixed delay</dt><dd>${fmt(v.delay_secs * 1000, 3)} ms</dd>
          <dt>Clock drift</dt><dd>${fmt(v.drift_ppm, 2)} ppm</dd>
          <dt>RMS residual</dt><dd>${fmt(v.rms_residual_secs * 1000, 3)} ms</dd>
          <dt>Max |residual|</dt><dd>${fmt(v.max_abs_residual_secs * 1000, 3)} ms</dd>
          <dt>Anchors</dt><dd>${v.anchors_used} used · ${v.anchors_rejected} rejected · ${v.anchors_unknown} unknown</dd>
        </dl>
      </div>`,
    )
    .join("");
}

function renderMapReport(view: RenderView): void {
  const rows = view.reports
    .map(
      (r) => `
      <tr>
        <td>${r.track_id}</td>
        <td>${r.output_frames} frames (${fmt(r.output_duration_secs, 3)} s)</td>
        <td>${r.head_pad_frames}</td>
        <td>${r.tail_pad_frames}</td>
        <td>${r.unknown_frames}</td>
        <td>${fmt(r.min_map_slope, 7)} … ${fmt(r.max_map_slope, 7)}</td>
      </tr>`,
    )
    .join("");
  $("map-view").innerHTML = `
    <p>All stems share one project length ending at
      <strong>${fmt(view.project_end_secs, 3)} s</strong>. Frames outside any
      mapped source sample are zero-filled; edge-clamped frames and unknown
      regions are counted below.</p>
    <table>
      <thead><tr>
        <th>Track</th><th>Output</th><th>Head pad</th><th>Tail pad</th>
        <th>Unknown frames</th><th>Map slope range</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function pickFile(): Promise<string | null> {
  // Tauri v2 dialog plugin would provide native pickers; keeping the command
  // surface generic lets the core stay independent of the frontend plugin set.
  const path = window.prompt("Path to WAV file:");
  return path && path.trim().length > 0 ? path.trim() : null;
}

function wire(): void {
  $("new-project").addEventListener("click", async () => {
    const name = window.prompt("Project name:", "rehearsal") ?? "rehearsal";
    await invoke("new_project", { name });
    tracks = [];
    anchors = [];
    referenceId = null;
    renderTracks();
    renderAnchors();
    renderFit([]);
    $("map-view").innerHTML = "";
  });

  $("import-track").addEventListener("click", async () => {
    const path = await pickFile();
    if (!path) return;
    const meta = await invoke<TrackMeta>("import_track", { path });
    tracks.push(meta);
    if (referenceId === null) referenceId = meta.id;
    renderTracks();
  });

  $("detect").addEventListener("click", async () => {
    anchors = await invoke<AnchorView[]>("run_detection");
    renderAnchors();
  });

  $("fit").addEventListener("click", async () => {
    const views = await invoke<FitView[]>("fit_all");
    anchors = views.flatMap((v) => v.anchors);
    renderAnchors();
    renderFit(views);
  });

  $("render").addEventListener("click", async () => {
    const dir = window.prompt("Output directory:", "aligned-out");
    if (!dir) return;
    const view = await invoke<RenderView>("render_aligned", { outputDir: dir });
    renderMapReport(view);
  });

  $("export-positions").addEventListener("click", async () => {
    const path = window.prompt("Positions CSV path:", "positions.csv");
    if (!path) return;
    const rows = await invoke<number>("export_positions", { path });
    window.alert(`Exported ${rows} anchor rows to ${path}`);
  });

  $("save-project").addEventListener("click", async () => {
    const path = window.prompt("Save project as:", "project.json");
    if (path) await invoke("save_project", { path });
  });

  $("load-project").addEventListener("click", async () => {
    const path = await pickFile();
    if (!path) return;
    const project = await invoke<import("./types").Project>("load_project", { path });
    referenceId = project.reference_track_id;
    tracks = project.tracks;
    anchors = project.anchors.map((a) => a as AnchorView);
    $("epoch-badge").textContent = `epoch ${project.epoch}`;
    renderTracks();
    renderAnchors();
  });
}

window.addEventListener("DOMContentLoaded", wire);
