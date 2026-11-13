# Helper scripts

- `copy_models_from_cluster.sh --port <port>`: copy model bundles, impulse
  records and notebook-job results from EIDF into `models/` and `output/`,
  then rebuild the registry index. See `docs/cluster_eidf.md`.
- `add_labels.sh`, `add_labels_nodal_ko.sh`, `add_labels_composite.sh`:
  `./add_labels.sh input.mp4 output.mp4` writes marker-name labels onto a
  video of NCA channels with ffmpeg. The channel names and positions are fixed
  in each script, and the font path is for Linux.
