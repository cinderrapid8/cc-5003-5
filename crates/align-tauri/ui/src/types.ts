export interface Project {
  version: number;
  name: string;
  tracks: import("./main").TrackMeta[];
  reference_track_id: string | null;
  epoch: number;
  anchors: import("./main").AnchorView[];
  maps: unknown[];
}
