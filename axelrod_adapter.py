from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import Any


def _axelrod_source_root() -> Path:
    return Path(__file__).resolve().parent / "Axelrod" / "axelrod"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_axelrod_core() -> tuple[Any, Any]:
    try:
        import axelrod as axl

        return axl.Action, axl.Game
    except Exception:
        source_root = _axelrod_source_root()
        action_module = _load_module("_pd_axelrod_action", source_root / "action.py")
        action_cls = action_module.Action

        # Reference implementation: https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/game.py
        # Adaptation note: load only Action/Game from the submodule source to avoid optional plotting deps.
        stub_module = types.ModuleType("axelrod")
        stub_module.Action = action_cls

        previous_axelrod_module = sys.modules.get("axelrod")
        sys.modules["axelrod"] = stub_module
        try:
            game_module = _load_module("_pd_axelrod_game", source_root / "game.py")
        finally:
            if previous_axelrod_module is None:
                del sys.modules["axelrod"]
            else:
                sys.modules["axelrod"] = previous_axelrod_module

        return action_cls, game_module.Game
