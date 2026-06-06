"""Smoke test — verifies every module imports and the top-level scripts parse.

Run from the repo root:

    python scripts/smoke_test.py

Does NOT need a trained model, dataset, or GPU. Useful as a post-install check
on both the Mac and the RTX box.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import traceback
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MODULES = [
    "football_tracker",
    "football_tracker.datasets.soccernet",
    "football_tracker.datasets.roboflow",
    "football_tracker.datasets.pitch_roboflow",
    "football_tracker.datasets.youtube",
    "football_tracker.datasets.merge",
    "football_tracker.training.train_detector",
    "football_tracker.training.train_pitch",
    "football_tracker.training.eval_detector",
    "football_tracker.training.eval_pitch",
    "football_tracker.training.eval_homography",
    "football_tracker.training.export",
    "football_tracker.tracking.bytetrack",
    "football_tracker.tracking.trail",
    "football_tracker.pitch.homography",
    "football_tracker.pitch.keypoints",
    "football_tracker.pitch.dynamic_homography",
    "football_tracker.pitch.minimap",
    "football_tracker.analytics.distance",
    "football_tracker.analytics.heatmap",
    "football_tracker.pipeline.live",
]

SCRIPTS = ["dataset.py", "train.py", "eval.py", "demo.py"]


def main() -> int:
    failed: list[tuple[str, str]] = []
    for mod in MODULES:
        try:
            importlib.import_module(mod)
            print(f"  ok    {mod}")
        except Exception as e:                     # noqa: BLE001
            print(f"  FAIL  {mod}: {e}")
            failed.append((mod, traceback.format_exc()))

    # Verify minimap renders without crashing
    try:
        import numpy as np
        from football_tracker.pitch.minimap import Minimap
        Minimap(width_px=200).render(np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32))
        print("  ok    minimap.render()")
    except Exception as e:                         # noqa: BLE001
        print(f"  FAIL  minimap.render(): {e}")
        failed.append(("minimap.render", traceback.format_exc()))

    # Verify pitch keypoint scheme loads and has the expected shape
    try:
        from football_tracker.pitch.keypoints import load as load_scheme
        scheme = load_scheme()
        assert scheme.num == 32, f"expected 32 keypoints, got {scheme.num}"
        assert scheme.world_xy.shape == (32, 2)
        assert len(scheme.flip_idx) == 32
        print(f"  ok    pitch_keypoints (n={scheme.num}, pitch={scheme.pitch_size_m})")
    except Exception as e:                         # noqa: BLE001
        print(f"  FAIL  pitch_keypoints: {e}")
        failed.append(("pitch_keypoints", traceback.format_exc()))

    # Verify each top-level script parses its --help
    repo_root = Path(__file__).resolve().parent.parent
    for script in SCRIPTS:
        path = repo_root / script
        try:
            r = subprocess.run(
                [sys.executable, str(path), "--help"],
                capture_output=True, text=True, timeout=15,
            )
            assert r.returncode == 0, r.stderr
            print(f"  ok    {script} --help")
        except Exception as e:                     # noqa: BLE001
            print(f"  FAIL  {script} --help: {e}")
            failed.append((script, traceback.format_exc()))

    if failed:
        print(f"\n{len(failed)} failures:")
        for name, tb in failed:
            print(f"\n--- {name} ---\n{tb}")
        return 1
    print("\nAll good ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
