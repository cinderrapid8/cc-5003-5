mod commands;

use commands::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            inner: std::sync::Mutex::new(commands::State {
                project: align_core::Project::new("untitled"),
                project_dir: std::path::PathBuf::new(),
                buffers: Vec::new(),
            }),
        })
        .invoke_handler(tauri::generate_handler![
            commands::new_project,
            commands::import_track,
            commands::set_reference,
            commands::run_detection,
            commands::pick_candidate,
            commands::set_manual_anchor,
            commands::mark_anchor_unknown,
            commands::fit_all,
            commands::render_aligned,
            commands::save_project,
            commands::load_project,
            commands::list_anchors,
            commands::export_positions,
        ])
        .run(tauri::generate_context!())
        .expect("error while running band-align");
}
