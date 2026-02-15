# antenna_fdtdx

FDTDX-side pipeline for rapid antenna exploration, threshold screening, and campaign automation.

This folder is the **search engine** of the project: it evaluates many candidate layouts quickly and produces ranked candidates for higher-fidelity openEMS validation.

## Conceptual model

The FDTDX pipeline treats the radiator region as an editable 2D copper field on top of a wedge-slice PCB model. It then computes port/matching and directional proxy metrics to guide search.

Core idea:
- Use FDTDX for throughput and gradient-friendly optimization loops.
- Use openEMS later as the validation source of truth.

## Folder structure and responsibility

- `sim/`
  - Simulation model definitions, geometry generation, and metric extraction.
  - Entry point: `python -m sim.run_one`.
- `tools/`
  - Campaign daemon and helper CLIs (calibration, threshold checks, reports, export).
- `calibration/`
  - Port reference and calibration JSON files used to stabilize port-derived metrics.
- `hybrid_seeds/`
  - Seed artifacts used by hybrid run generation logic.
- `campaign_queue.json`
  - Queue template describing run matrix, thresholds, and openEMS bridge behavior.
- `campaign_state.json`
  - Runtime state written by the daemon (generated, not hand-edited).

## Important tools and how they work

### `sim.run_one`

Runs one FDTDX simulation and writes a run folder.

What it does internally:
1. Loads a model from `sim/models.py`.
2. Builds geometry from an init mode or `--params` override.
3. Runs FDTDX field solve and extracts port quantities.
4. Computes sweep summaries and matching metrics.
5. Writes output artifacts (`metrics.json`, `summary.txt`, `s11.csv`, `geometry.npz`, etc).

Typical command:

```bash
cd antenna_fdtdx
python -m sim.run_one --model dir5 --quality coarse --backend cpu --init patch --seed 0
```

### `tools.campaign_daemon`

Queue orchestrator for large multi-run campaigns.

How it works:
1. Reads queue config (`--queue`) and runtime state (`--state`).
2. Schedules pending runs across available GPUs.
3. Launches optimizer jobs (via `python -m opt.optimize_topology ...` when present in your branch).
4. Periodically triggers threshold/CPU checks.
5. Optionally triggers openEMS validation jobs for selected candidates.
6. Appends machine-readable events/results into a JSONL log.

Notes:
- The daemon is resilient to restarts because state is persisted in JSON.
- Queue sections (`defaults`, `cpu_threshold`, `openems`, `hybrid`) each control a part of the lifecycle.

### `tools.threshold_watch`

Evaluates thresholded/binary design snapshots at intervals and writes CPU-side metrics. Useful for detecting "good in soft space, bad after binarization" regressions.

### `tools.port_reference`

Creates open/short reference runs for a model and writes a port reference JSON. This is used by Thevenin-like metric paths and helps stabilize comparisons.

### `tools.port_calibration`

Builds FDTDX→openEMS calibration mapping for port S11 metrics.

### `tools.baseline_suite`

Runs baseline sanity cases (and optionally delegates to `antenna_opt` baselines) to verify that key electrical behaviors remain aligned.

### `tools.run_openems_from_fdtdx`

Thin wrapper that forwards to `antenna_opt.tools.run_openems_from_fdtdx` so openEMS validation can be triggered from the FDTDX side.

### `tools.export_geometry`

Converts run geometry into SVG copper outlines (with optional thresholding and feed inclusion).

### `tools.smooth_copper_svg`

Post-processes pixelated copper SVGs into smoother manufacturable contours and can align top/ground overlays.

### `tools.report_top_runs`

Aggregates top FDTDX runs and maps them to any matching openEMS validation runs.

### `tools.archive_passes`

Copies/symlinks runs that pass acceptance filters into a curated archive folder.

### `tools.plot_smith`

Plots Smith chart visualizations from saved CSV sweep data.

## Queue file (`campaign_queue.json`) guide

Key sections:
- `runs`: explicit run list (model/init/seed triplets).
- `defaults`: shared arguments applied to each run unless overridden.
- `cpu_threshold`: threshold evaluation cadence and stopping rules.
- `openems`: validation policy, concurrency, and bridge arguments.
- `hybrid`: optional seed-combination logic.
- `results_log`: JSONL event log path.

Recommended pattern:
1. Keep `campaign_queue.json` as a portable template (relative paths).
2. Generate machine-local resolved config via `../scripts/configure_queue.sh`.
3. Run daemon against `campaign_queue.local.json`.

## Output artifacts

Depending on workflow stage, generated artifacts typically include:
- `params.json`: run inputs and model metadata.
- `metrics.json`: numerical summary for ranking/filtering.
- `summary.txt`: human-readable metric summary.
- `s11.csv` and optional plots.
- `geometry.npz`: serialized geometry tensors/grids.
- `history.jsonl`, best snapshots, threshold eval files (for optimization flows).

## Recommended operating sequence

1. Run `tools.port_reference` / `tools.port_calibration` when model or port geometry changes.
2. Run small baseline suite to confirm no obvious extraction breakage.
3. Start campaign daemon for broad search.
4. Use threshold checks to filter unstable binary candidates.
5. Validate top candidates with openEMS before drawing conclusions.

## Known tradeoff

FDTDX metrics are optimized for speed and ranking signal. Treat openEMS validation (`../antenna_opt`) as the final decision gate for publishable RL/gain/FB claims.

## License

MIT. See `LICENSE`.
