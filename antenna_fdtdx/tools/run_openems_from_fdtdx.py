"""Thin wrapper to run antenna_opt's openEMS bridge from antenna_fdtdx."""

from __future__ import annotations

import importlib.util
import os
import sys


def _load_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    antenna_opt_root = os.path.join(repo_root, "antenna_opt")
    target = os.path.join(antenna_opt_root, "tools", "run_openems_from_fdtdx.py")
    if not os.path.isfile(target):
        raise FileNotFoundError(f"Missing openEMS bridge at {target}")
    if antenna_opt_root not in sys.path:
        sys.path.insert(0, antenna_opt_root)
    tools_init = os.path.join(antenna_opt_root, "tools", "__init__.py")
    if not os.path.isfile(tools_init):
        raise FileNotFoundError(f"Missing antenna_opt tools package at {tools_init}")
    original_tools = sys.modules.get("tools")
    tools_spec = importlib.util.spec_from_file_location("tools", tools_init)
    if tools_spec is None or tools_spec.loader is None:
        raise RuntimeError(f"Failed to load antenna_opt tools spec for {tools_init}")
    tools_module = importlib.util.module_from_spec(tools_spec)
    tools_spec.loader.exec_module(tools_module)
    sys.modules["tools"] = tools_module
    spec = importlib.util.spec_from_file_location("antenna_opt_run_openems_from_fdtdx", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {target}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if original_tools is not None:
            sys.modules["tools"] = original_tools
        else:
            sys.modules.pop("tools", None)
    return module


def main() -> int:
    module = _load_module()
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
