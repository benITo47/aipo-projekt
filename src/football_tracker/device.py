"""Pick the best inference device available — CUDA > MPS (Apple Silicon) > CPU.

`pick_device()` returns an ultralytics-style device string:

    Input                              Result
    ─────────────────────────────────  ──────────────────────────────
    None / "auto"                      CUDA(0) > MPS > CPU
    "cuda"                             "0" if CUDA available, else SystemExit
    "cuda:N" / "N"                     "N" if CUDA available, else SystemExit
    "0,1,2"                            multi-GPU string (CUDA only)
    "mps"                              "mps" on Apple Silicon, else SystemExit
    "cpu"                              "cpu"

This is what your friend on the RTX box uses to pin a specific GPU
(`--device 0` or `--device cuda:1`); on the Mac it'll auto-pick MPS.
"""
from __future__ import annotations


def pick_device(preferred: str | None = None) -> str:
    import torch

    cuda_ok = torch.cuda.is_available()
    mps_ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    # Auto
    if not preferred or preferred == "auto":
        if cuda_ok:
            return "0"
        if mps_ok:
            return "mps"
        return "cpu"

    pref = preferred.strip().lower()

    if pref == "cpu":
        return "cpu"

    if pref == "mps":
        if not mps_ok:
            raise SystemExit("MPS requested but not available on this machine.")
        return "mps"

    if pref == "cuda":
        if not cuda_ok:
            raise SystemExit("CUDA requested but not available on this machine.")
        return "0"

    # "cuda:N" → "N"
    if pref.startswith("cuda:"):
        pref = pref.removeprefix("cuda:")

    # At this point: "0", "1", "0,1,2", or junk. Anything CUDA needs CUDA.
    if not cuda_ok:
        raise SystemExit(
            f"Device '{preferred}' looks like a CUDA index but CUDA is not available."
        )
    return pref
