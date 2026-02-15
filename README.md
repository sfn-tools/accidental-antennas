# Accidental Antennas

Tool to brute-force PCB antenna designs using GPUs.

This repository combines a fast GPU exploration path (FDTDX) with a higher-fidelity validation path (openEMS), then adds tooling to automate long multi-run campaigns.

## What this project does

- Generates candidate PCB antenna geometries on a 2D copper grid.
- Evaluates matching and directional behavior quickly in FDTDX.
- Re-checks promising candidates in openEMS before treating them as real results.
- Exports copper geometry for downstream PCB CAD workflows.

In short: **search fast, validate carefully, then export**.

## High-level architecture

1. `antenna_fdtdx` runs many candidate designs and writes per-run metrics/geometry.
2. Threshold/CPU checks screen binary designs and filter obvious failures.
3. `antenna_opt` re-simulates selected candidates in openEMS (NF2FF, gain, FB, impedance).
4. Archive/report utilities collect the designs that pass your acceptance criteria.

## Repository layout

| Path | Purpose | License |
| --- | --- | --- |
| `antenna_fdtdx/` | FDTDX-side simulation/orchestration, threshold checks, calibration, reporting | MIT |
| `antenna_opt/` | openEMS-side simulation, baseline validation, FDTDX bridge, geometry export | MIT |
| `scripts/` | Bootstrap/environment/queue helper scripts for local automation | MIT |
| `fdtdx/` | Upstream FDTDX source mirror/vendor code | Upstream (`fdtdx/LICENSE`) |
| `openEMS-Project/` | Upstream openEMS/CSXCAD source mirror/vendor code | Upstream (project-provided licenses) |
| `manifests/` | Reproducibility manifests (packages and commit pins) | Metadata |

## End-to-end workflow

### 1) Set up runtime

```bash
./scripts/bootstrap.sh
source ./scripts/env.sh
```

What happens:
- `bootstrap.sh` installs system dependencies, creates `.venv`, installs Python packages, builds openEMS under `opt/openEMS`, and writes `antenna_fdtdx/campaign_queue.local.json`.
- `env.sh` exports runtime variables (`OPENEMS_ROOT`, `PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`) and activates `.venv` if present.

### 2) Run exploration

- One-off run: `python -m sim.run_one` from `antenna_fdtdx`.
- Batch queue: `./scripts/run_daemon.sh` (starts `antenna_fdtdx.tools.campaign_daemon`).

### 3) Validate in openEMS

- Use `antenna_opt.tools.run_openems_from_fdtdx` (or the thin wrapper in `antenna_fdtdx/tools/run_openems_from_fdtdx.py`) to replay selected FDTDX candidates in openEMS.

### 4) Inspect and export

- Metrics and summaries are written per run.
- Geometry can be exported/smoothed to SVG for manufacturing workflows.

## Where results live

Typical generated folders (ignored by `.gitignore`):
- `antenna_fdtdx/runs*`, `antenna_fdtdx/opt_runs*`, `antenna_fdtdx/runs_threshold*`
- `antenna_opt/runs*`, `antenna_opt/runs_fdtdx*`
- `antenna_fdtdx/passing_designs`

Per-run artifacts usually include:
- `params.json`
- `metrics.json`
- `summary.txt`
- frequency responses (`s11.csv`, plots)
- geometry payload (`geometry.npz` on FDTDX side)

## Documentation map

- FDTDX workflow and tool inventory: `antenna_fdtdx/README.md`
- openEMS workflow and validation tools: `antenna_opt/README.md`
- automation/bootstrap scripts: `scripts/README.md`
- license boundaries: `LICENSES.md`

## Licensing

- `antenna_fdtdx`, `antenna_opt`, and `scripts` are MIT licensed.
- `fdtdx` and `openEMS-Project` remain under their upstream licenses.
- See `LICENSES.md` for the exact split.
