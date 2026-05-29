"""Shared utilities for NN-vs-GZ kernel optimization and data preparation."""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RS = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
PRESCREEN_RS = [1.0, 2.0, 5.0]
DEFAULT_HIDDEN = [16, 16]

NN_FUNC_KEY = "NN@model.json"
NN_NK_STEM = "NN_model.json"
GZ_TARGETS_FILE = "gz_targets.npz"
MANIFEST_FILE = "manifest.json"
POWER_SWEEP_FILE = "power_sweep.json"


class Log:
    def __init__(self, quiet: bool = False, verbose: bool = False) -> None:
        self.quiet = quiet
        self.show_cpp_stdout = verbose

    def info(self, msg: str = "") -> None:
        if not self.quiet:
            print(msg, flush=True)

    def detail(self, msg: str = "") -> None:
        if not self.quiet:
            print(msg, flush=True)

    def verbose(self, msg: str = "") -> None:
        if self.show_cpp_stdout and not self.quiet:
            print(msg, flush=True)


def format_per_rs(per: dict[float, float]) -> str:
    return "  ".join(f"r_s={rs:g}:{per[rs]:.5f}" for rs in sorted(per))


def run_dump_gz(
    dump_bin: Path,
    rs_list: list[float],
    nk: int,
    kmax: float,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Return {r_s: (k_over_kf, n_gz)} from dump_gz_grid."""
    rs_arg = ",".join(f"{r:g}" for r in rs_list)
    proc = subprocess.run(
        [str(dump_bin), "--rs", rs_arg, "--n", str(nk), "--kmax", str(kmax)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    by_rs: dict[float, list[tuple[float, float]]] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rs, x, n = map(float, line.split())
        by_rs.setdefault(rs, []).append((x, n))
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for rs, rows in by_rs.items():
        arr = np.array(rows)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        out[rs] = (arr[:, 0], arr[:, 1])
    return out


def gz_dict_to_arrays(
    gz: dict[float, tuple[np.ndarray, np.ndarray]],
    rs_list: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack {rs: (k, n)} into aligned (rs, k, n_gz) arrays; k grid from first rs."""
    rs_arr = np.array(rs_list, dtype=float)
    k_ref = gz[rs_list[0]][0]
    n_rows = []
    for rs in rs_list:
        k, n = gz[rs]
        if len(k) != len(k_ref) or np.max(np.abs(k - k_ref)) > 1e-12:
            n_rows.append(np.interp(k_ref, k, n))
        else:
            n_rows.append(n)
    return rs_arr, k_ref, np.vstack(n_rows)


def save_gz_targets(
    path: Path,
    gz: dict[float, tuple[np.ndarray, np.ndarray]],
    rs_list: list[float],
    *,
    kmax: float,
    nk: int,
    weight_x2: bool,
) -> None:
    rs_arr, k, n_gz = gz_dict_to_arrays(gz, rs_list)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rs=rs_arr,
        k=k,
        n_gz=n_gz,
        kmax=np.array(kmax),
        nk=np.array(nk),
        weight_x2=np.array(1 if weight_x2 else 0),
    )


def load_gz_targets(path: Path) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    data = np.load(path)
    rs_list = [float(x) for x in data["rs"]]
    k = np.asarray(data["k"], dtype=float)
    n_gz = np.asarray(data["n_gz"], dtype=float)
    return {rs: (k, n_gz[i]) for i, rs in enumerate(rs_list)}


def ensure_gz_targets(
    data_dir: Path,
    dump_bin: Path,
    rs_list: list[float],
    nk: int,
    kmax: float,
    weight_x2: bool,
    log: Log,
    *,
    refresh: bool = False,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Load gz_targets.npz from data_dir or build it via dump_gz_grid."""
    data_dir.mkdir(parents=True, exist_ok=True)
    npz = data_dir / GZ_TARGETS_FILE
    manifest_path = data_dir / MANIFEST_FILE

    if npz.is_file() and not refresh:
        gz = load_gz_targets(npz)
        missing = [rs for rs in rs_list if rs not in gz]
        if not missing:
            log.info(f"Loaded GZ targets from {npz} ({len(rs_list)} r_s)")
            return {rs: gz[rs] for rs in rs_list}
        log.info(f"Cache missing r_s {missing}; refreshing {npz}")

    log.info("Building GZ reference n(k) via dump_gz_grid...")
    t0 = time.time()
    gz = run_dump_gz(dump_bin, rs_list, nk, kmax)
    save_gz_targets(npz, gz, rs_list, kmax=kmax, nk=nk, weight_x2=weight_x2)
    manifest = {
        "rs": rs_list,
        "kmax": kmax,
        "gz_n": nk,
        "weight_x2": weight_x2,
        "created_unix": time.time(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"  wrote {npz} ({time.time() - t0:.1f}s)")
    return gz


def rmse_vs_gz(
    k_m: np.ndarray,
    n_m: np.ndarray,
    gz_cache: dict[float, tuple[np.ndarray, np.ndarray]],
    rs: float,
    *,
    kmax: float,
    nk: int,
    weight_x2: bool,
) -> float:
    x_gz, n_gz = gz_cache[rs]
    x = np.linspace(0.0, kmax, nk)
    n_ref = np.interp(x, x_gz, n_gz)
    n_mod = np.interp(x, k_m, n_m)
    w = x * x if weight_x2 else np.ones_like(x)
    diff = n_mod - n_ref
    mse = float(_trapz(w * diff * diff, x) / _trapz(w, x))
    return math.sqrt(mse)


def l1_error_vs_gz(
    k_m: np.ndarray,
    n_m: np.ndarray,
    gz_cache: dict[float, tuple[np.ndarray, np.ndarray]],
    rs: float,
    *,
    kmax: float,
    nk: int,
) -> float:
    """Per-r_s L1 deviation against the GZ reference.

        E(r_s) = int_0^{kmax * k_F} |n(k) - n_ref(k)| dk            (Hartree units)

    Implemented in the dimensionless variable x = k / k_F (which is the
    convention used throughout the repository: ``load_nk_tsv`` already
    divides the k column by k_F).  ``kmax`` is the upper limit in those
    units, so passing ``kmax = 3`` integrates to k = 3 k_F.  The k_F
    factor that would convert this to absolute Hartree units cancels out
    of the loss because the optimizer only compares relative values, and
    the dimensionless integral has a uniform scale across r_s
    (0 <= E <= kmax for n, n_ref in [0, 1]).
    """
    x_gz, n_gz = gz_cache[rs]
    x = np.linspace(0.0, kmax, nk)
    n_ref = np.interp(x, x_gz, n_gz)
    n_mod = np.interp(x, k_m, n_m)
    return float(_trapz(np.abs(n_mod - n_ref), x))


def aggregate_rmse(per_rs: dict[float, float]) -> float:
    """RMSE of per-r_s L1 errors:  sqrt( mean_i E_i^2 ).

    The NN is trained to minimise this scalar.  Non-finite per-r_s values
    propagate to the aggregate (returned as +inf), which is the natural
    signal for a crashed SCF or missing n(k) export.  No UNCONV_PENALTY
    clamp is applied here: gradient-free optimizers can still make
    progress because the C++ driver writes n(k) even on non-convergence
    (see ``main.cpp``), so the achieved L1 error is finite for any solve
    that completed at all.
    """
    if not per_rs:
        return float("inf")
    vals = list(per_rs.values())
    if any(not math.isfinite(v) for v in vals):
        return float("inf")
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def load_nk_tsv(path: Path) -> tuple[np.ndarray, np.ndarray, bool | None]:
    k_list: list[float] = []
    n_list: list[float] = []
    converged: bool | None = None
    kf = 1.0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("# kF="):
                try:
                    kf = float(line.split("kF=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("# converged:"):
                try:
                    converged = int(float(line.split(":")[1].strip())) == 1
                except ValueError:
                    converged = None
            elif line.startswith("#") or not line:
                continue
            else:
                parts = line.split()
                if len(parts) >= 2:
                    k_list.append(float(parts[0]))
                    n_list.append(float(parts[1]))
    k = np.array(k_list)
    n = np.array(n_list)
    if kf > 0:
        k = k / kf
    return k, n, converged
