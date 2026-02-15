# scripts

Repository-level operational helpers for setup and queue execution.

These scripts are small, but they define the expected runtime shape for the rest of the project.

## Script inventory

### `bootstrap.sh`

Bootstraps a fresh machine for this repository.

What it does:
1. Installs system packages (Debian/Ubuntu via `apt-get`).
2. Creates local Python venv at `./.venv`.
3. Installs Python dependencies (`manifests/requirements_base.txt`) and editable `fdtdx`.
4. Builds openEMS under `./opt/openEMS` using `openEMS-Project/update_openEMS.sh`.
5. Generates a machine-local queue file via `configure_queue.sh`.

Use this once per new environment, then update only as needed.

### `env.sh`

Sets runtime environment variables and activates `.venv`.

Exports:
- `OPENEMS_ROOT`
- `PATH` (prepends `OPENEMS_ROOT/bin`)
- `LD_LIBRARY_PATH` (prepends `OPENEMS_ROOT/lib`)
- `PYTHONPATH` (prepends `antenna_fdtdx` and `antenna_opt`)

Most Python commands in this project assume this environment is already active.

### `configure_queue.sh`

Transforms `antenna_fdtdx/campaign_queue.json` into `antenna_fdtdx/campaign_queue.local.json`.

What it modifies:
- resolves local run roots for threshold/openEMS outputs,
- detects visible GPUs via `nvidia-smi`,
- assigns `gpu` indices across queued runs (round-robin),
- removes GPU assignments when no GPU is available.

Use this whenever queue structure or available GPU count changes.

### `run_daemon.sh`

Starts `antenna_fdtdx.tools.campaign_daemon` with local queue.

Behavior:
- loads environment via `env.sh`,
- ensures `campaign_queue.local.json` exists,
- launches daemon with `nohup` in background,
- writes daemon logs under `antenna_fdtdx/`.

## Standard usage

```bash
# from repository root
./scripts/bootstrap.sh
source ./scripts/env.sh
./scripts/configure_queue.sh
./scripts/run_daemon.sh
```

## Operational notes

- `bootstrap.sh` is intentionally opinionated for Linux workstations and build servers.
- The queue template should stay portable; machine-specific paths belong in `campaign_queue.local.json`.
- If openEMS binaries move, set `OPENEMS_ROOT` before sourcing `env.sh`.

## Troubleshooting checklist

- If Python cannot import `openEMS`/`CSXCAD`, re-source `env.sh` and verify `OPENEMS_ROOT`.
- If runs do not schedule on expected GPUs, rerun `configure_queue.sh` and inspect generated `gpu` fields.
- If daemon appears idle, inspect `antenna_fdtdx/campaign_daemon.log` and `antenna_fdtdx/campaign_state.json`.

## License

MIT. See `LICENSE`.
