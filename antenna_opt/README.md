# antenna_opt

openEMS-side simulation, validation, and export toolkit.

This folder is the **validation engine** of the project. It is used both for standalone openEMS model work and for replaying FDTDX-generated geometries under a more trusted electromagnetic solver configuration.

## What this folder is responsible for

- Running openEMS models for wedge-slice antennas.
- Producing S11/impedance outputs and optional NF2FF-derived metrics.
- Bridging FDTDX run folders into openEMS-compatible geometry and ports.
- Exporting copper geometry assets and pattern visualizations.

## Folder structure

- `sim/`
  - openEMS model definitions and `sim.run_one` execution path.
- `tools/`
  - baseline sweeps, FDTDX bridge runner, sensitivity scan, geometry export, pattern visualization.
- `designs/`
  - hand-tuned/reference parameter sets and geometric constraints.
- `exports/`
  - generated geometry SVG/JSON artifacts.
- `reports/`
  - baseline and campaign summaries/diagnostics.
- `tests/`
  - smoke checks and parameter-bound tests.

## Core tools and behavior

### `sim.run_one`

Runs one openEMS simulation for a chosen model and parameter set.

What it writes:
- `params.json`
- `metrics.json`
- `summary.txt`
- `s11.csv`, `s11.png`
- optional matched outputs (`s11_matched*`)
- optional pattern CSVs / NF2FF metadata

Example:

```bash
cd antenna_opt
python -m sim.run_one --model dir5 --quality fast
```

### `tools.run_openems_from_fdtdx`

Main bridge for validating FDTDX candidates.

How it works:
1. Reads FDTDX run artifacts (`params.json`, `geometry.npz`).
2. Applies thresholding and optional geometry smoothing/meshing controls.
3. Configures openEMS port behavior (`--port-mode`, `--port-type`).
4. Runs openEMS solve.
5. Writes full validation run folder under `runs_fdtdx/*` by default.
6. Optionally computes NF2FF metrics (`--metrics`) and VTK outputs.

This is the command to use when confirming whether a candidate survives outside the FDTDX approximation stack.

### `tools.baseline_suite`

Regression/sanity driver for canonical cases (`microstrip`, `patch`, `plate`).

Use this after changing mesh, port setup, or material assumptions to make sure S-parameter behavior has not drifted unexpectedly.

### `tools.export_geometry`

Exports model-generated top/ground polygons to SVG plus dimensions JSON for CAD handoff.

### `tools.sensitivity_scan`

Perturbs model parameters and reports metric sensitivity, useful when diagnosing which parameters are doing most resonance/matching work.

### `tools.visualize_pattern`

Converts NF2FF result data from a run into a VTK file for 3D visualization (e.g., ParaView).

## Quality levels and tradeoffs

Typical quality modes: `fast`, `medium`, `high`.

Practical use:
- `fast`: quick iteration and broad sweeps.
- `medium`: normal validation pass.
- `high`: expensive spot-checks before reporting final numbers.

Always compare results at equivalent quality settings when making claims about relative performance.

## Typical validation workflow

1. Pick a candidate run from `../antenna_fdtdx/runs*`.
2. Run `tools.run_openems_from_fdtdx` with explicit port mode/type and threshold.
3. Inspect `metrics.json` and `s11.csv` in the output run folder.
4. If needed, rerun with tighter mesh/time settings.
5. Keep only candidates that remain good after thresholding and openEMS replay.

## Environment assumptions

- `openEMS` and `CSXCAD` Python modules are importable.
- The runtime environment has `numpy`, `scipy`, `matplotlib`, and `h5py`.
- Recommended bootstrap path from repo root:

```bash
source ../scripts/env.sh
```

## Convenience scripts

- `run_openems_sweep.sh`: evaluate one selected/latest FDTDX run across a fixed port/overlap matrix.
- `run_openems_sweep_latest.sh`: repeat sweeps over the latest N runs.
- `run_openems_sweep_validate.sh`: validation-oriented sweep over selected/latest run IDs.

These are wrappers around `tools.run_openems_from_fdtdx` to reduce CLI repetition.

## Why this stage matters

FDTDX is useful for search velocity, but openEMS validation is the reliability gate. Any candidate that does not reproduce here should be treated as non-viable for final reporting.

## License

MIT. See `LICENSE`.
