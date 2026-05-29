#!/usr/bin/env python3
"""Optimize separable NN kernel f(n) so RDMFT n(k, r_s) matches GZ reference.

Writes ``model.json`` and runs ``rdmft_heg --funcs NN@<json>`` in a subprocess
loop (zeroth-order optimization via SciPy).

Flow (default):
  0) ``make prepare-nn-data`` — cache GZ targets (and optional Power sweep) under
     ``build/nn_data/``.
  1) Prescreen — coarse r_s / grid, pick best of Power/Müller/random inits.
  2) Main — L-BFGS-B or Powell on flattened weights vs GZ n(k).
  3) Final — full solve + ``--nk-out`` under ``build/nn_best/``.

Requires NumPy and SciPy.  Build ``build/rdmft_heg`` and ``build/dump_gz_grid`` first.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nn_gz_common import (  # noqa: E402
    DEFAULT_HIDDEN,
    DEFAULT_RS,
    Log,
    NN_FUNC_KEY,
    NN_NK_STEM,
    PRESCREEN_RS,
    POWER_SWEEP_FILE,
    REPO_ROOT,
    UNCONV_PENALTY,
    _trapz,
    aggregate_rmse,
    ensure_gz_targets,
    tail_rmse_vs_gz,
    format_per_rs,
    load_nk_tsv,
)

try:
    from scipy import optimize as sco
except ImportError as exc:
    raise SystemExit(
        "optimize_nn_gz.py requires SciPy.  Install: pip install -r scripts/requirements-nn.txt"
    ) from exc


def softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 30.0, x, np.log1p(np.exp(np.clip(x, -30.0, 30.0))))


def eval_f(model: dict, n: float | np.ndarray) -> np.ndarray:
    """f(n) = n * softplus(raw(n)) with tanh hidden layers."""
    n_arr = np.atleast_1d(np.asarray(n, dtype=float))
    h = n_arr.copy()
    for layer in model["kernels"][:-1]:
        W = np.asarray(layer["W"], dtype=float)
        b = np.asarray(layer["b"], dtype=float)
        h = np.tanh(h @ W.T + b) if h.ndim == 1 and h.shape[0] != 1 else np.tanh(
            np.asarray(h, dtype=float) @ W.T + b
        )
    last = model["kernels"][-1]
    W = np.asarray(last["W"], dtype=float)
    b = np.asarray(last["b"], dtype=float)
    if h.ndim == 1:
        raw = (h @ W.T + b).reshape(-1)
    else:
        raw = h @ W.T + b
    raw = np.asarray(raw, dtype=float).reshape(-1) + float(model.get("out_bias", 0.0))
    return (n_arr.reshape(-1) * softplus(raw)).reshape(n_arr.shape)


def build_model(
    hidden: list[int],
    weights: list[np.ndarray] | None = None,
    *,
    kernel_type: str = "separable",
) -> dict:
    """Construct an MLP model dict.

    kernel_type:
      ``separable``  - input dim 1, kernel K(n_i, n_j) = f(n_i) f(n_j) with
                       f(n) = n * softplus(MLP(n)).  Loaded by NNFunctional.
      ``pair``       - input dim 2, kernel K(n_i, n_j) =
                       sqrt(n_i n_j) * softplus(MLP([n_i+n_j, n_i n_j])).
                       Loaded by NNPairFunctional (non-separable).
    """
    if kernel_type not in ("separable", "pair"):
        raise ValueError(f"unknown kernel_type: {kernel_type}")
    in0 = 1 if kernel_type == "separable" else 2
    sizes = [in0] + hidden + [1]
    kernels: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for ell in range(len(sizes) - 1):
        in_d, out_d = sizes[ell], sizes[ell + 1]
        if weights is not None and ell < len(weights):
            W = weights[ell]
        else:
            W = rng.normal(scale=0.1, size=(out_d, in_d))
        b = np.zeros(out_d)
        kernels.append(
            {
                "in": in_d,
                "out": out_d,
                "W": W.tolist(),
                "b": b.tolist(),
            }
        )
    return {
        "version": 1,
        "kernel_type": kernel_type,
        "name": "NN" if kernel_type == "separable" else "NNPair",
        "hidden": hidden,
        "kernels": kernels,
        "out_bias": 0.0,
    }


def init_pair_model(
    target: str = "hf",
    hidden: list[int] | None = None,
    alpha: float = 1.0,
) -> dict:
    """Initialize a 2-input pair MLP near a known kernel for stable warm-start.

    With K(a, b) = sqrt(a b) * softplus(raw), choosing
    softplus(raw) = (a b)^{(alpha - 1/2)} gives K(a, b) = (a b)^alpha.  We fit
    raw to that on a few representative pair anchors, leaving small random
    hidden weights so the optimizer can escape if needed.

    target:
      ``hf``      - alpha = 1   (Hartree-Fock pair kernel a*b)
      ``mueller`` - alpha = 1/2 (sqrt(a*b))
      ``power``   - alpha given explicitly via ``alpha``
    """
    if hidden is None:
        hidden = [8, 8]
    model = build_model(hidden, kernel_type="pair")
    if target == "hf":
        a = 1.0
    elif target == "mueller":
        a = 0.5
    elif target == "power":
        a = float(alpha)
    else:
        raise ValueError(f"unknown pair init target: {target}")
    anchors = np.array(
        [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.4, 0.7), (0.1, 0.9)], dtype=float
    )
    p = anchors[:, 0] * anchors[:, 1]
    sp_target = np.power(np.clip(p, 1e-8, None), a - 0.5)
    raw_tgt = np.log(np.expm1(np.clip(sp_target, 1e-8, 50.0)))
    model["out_bias"] = float(np.mean(raw_tgt))
    rng = np.random.default_rng(123)
    for layer in model["kernels"][:-1]:
        nin = layer["in"]
        nout = layer["out"]
        layer["W"] = (rng.normal(scale=0.05, size=(nout, nin))).tolist()
    return model


def init_power_model(alpha: float = 0.55, hidden: list[int] | None = None) -> dict:
    """Initialize MLP so f(n) ≈ n^alpha on a few anchor points."""
    if hidden is None:
        hidden = [8, 8]
    model = build_model(hidden)
    anchors = np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95])
    targets = np.power(anchors, alpha)
    raw_tgt = targets / anchors
    raw_tgt = np.log(np.expm1(np.clip(raw_tgt, 1e-8, 50.0)))
    # Set last-layer bias to mean raw at anchors (single-output head).
    model["out_bias"] = float(np.mean(raw_tgt))
    # Small random hidden weights keep SCF stable while allowing optimization.
    rng = np.random.default_rng(42)
    for layer in model["kernels"][:-1]:
        nin = layer["in"]
        nout = layer["out"]
        layer["W"] = (rng.normal(scale=0.05, size=(nout, nin))).tolist()
    return model


def write_model_json(path: Path, model: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(model, f, indent=2)


def pack_params(model: dict) -> np.ndarray:
    parts: list[np.ndarray] = []
    for layer in model["kernels"]:
        parts.append(np.asarray(layer["W"], dtype=float).ravel())
        parts.append(np.asarray(layer["b"], dtype=float).ravel())
    parts.append(np.array([float(model.get("out_bias", 0.0))]))
    return np.concatenate(parts)


def unpack_params(model: dict, theta: np.ndarray) -> dict:
    m = json.loads(json.dumps(model))  # deep copy
    idx = 0
    for layer in m["kernels"]:
        W = np.asarray(layer["W"], dtype=float)
        b = np.asarray(layer["b"], dtype=float)
        nw = W.size
        nb = b.size
        layer["W"] = theta[idx : idx + nw].reshape(W.shape).tolist()
        idx += nw
        layer["b"] = theta[idx : idx + nb].reshape(b.shape).tolist()
        idx += nb
    m["out_bias"] = float(theta[idx])
    return m


def nk_stem_for_key(key: str) -> str:
    """Match main.cpp ``nk_stem_for`` (@, ;, /, space -> _)."""
    if key == NN_FUNC_KEY:
        return NN_NK_STEM
    s = key
    for ch, rep in (("@", "_"), (";", "_"), ("/", "_"), (" ", "_")):
        s = s.replace(ch, rep)
    if s.endswith(".tsv"):
        s = s[:-4]
    return s


def load_power_sweep_init_alpha(data_dir: Path) -> float | None:
    path = data_dir / POWER_SWEEP_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        alpha = data.get("best_alpha")
        return float(alpha) if alpha is not None else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def run_rdmft_nk(
    exe: Path,
    work_dir: Path,
    rs_list: list[float],
    nk_out: Path | None = None,
    *,
    n_grid: int,
    kmax: float,
    init_uniform: float | None,
    log: Log,
    label: str = "",
) -> None:
    """Run rdmft_heg with ``work_dir/model.json`` and cwd=work_dir.

    Uses the stable CLI key ``NN@model.json`` so nk exports are always
    ``NN_model.json_rs<rs>.tsv`` regardless of temp directory paths.
    """
    work_dir = work_dir.resolve()
    if not (work_dir / "model.json").is_file():
        raise FileNotFoundError(f"expected {work_dir / 'model.json'}")
    if nk_out is None:
        nk_out = work_dir / "nk"
    nk_out = nk_out.resolve()
    nk_out.mkdir(parents=True, exist_ok=True)
    exe_abs = exe.resolve()
    cmd = [
        str(exe_abs),
        "--funcs",
        NN_FUNC_KEY,
        "--rs",
        ",".join(f"{r:g}" for r in rs_list),
        "--N",
        str(n_grid),
        "--kmax",
        str(kmax),
        "--nk-out",
        str(nk_out),
        "--force",
        "--out-dir",
        str(nk_out / "_energy_stub"),
    ]
    if init_uniform is not None:
        cmd.extend(["--init-uniform", str(init_uniform)])
    prefix = f"[{label}] " if label else ""
    log.detail(
        f"{prefix}rdmft_heg: N={n_grid} kmax={kmax} r_s=[{','.join(f'{r:g}' for r in rs_list)}]"
    )
    log.verbose(f"{prefix}cwd: {work_dir}")
    log.verbose(f"{prefix}cmd: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        log.info(f"{prefix}rdmft_heg FAILED ({elapsed:.1f}s)")
        if proc.stderr:
            log.info(proc.stderr.rstrip())
        raise RuntimeError(
            f"rdmft_heg failed (code {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
        )
    log.detail(f"{prefix}rdmft_heg done ({elapsed:.1f}s)")
    if log.show_cpp_stdout and proc.stdout:
        log.verbose(proc.stdout.rstrip())


def gz_rmse(
    exe: Path,
    dump_bin: Path,
    rs_list: list[float],
    gz_cache: dict[float, tuple[np.ndarray, np.ndarray]],
    *,
    n_grid: int,
    kmax: float,
    nk: int,
    weight_x2: bool,
    init_uniform: float | None,
    work_dir: Path,
    log: Log,
    label: str = "",
    loss_mode: str = "sum",
    tail_weight: float = 0.0,
    tail_power: float = 4.0,
    tail_cutoff: float = 1.5,
) -> tuple[float, dict[float, float]]:
    """Run SCF and return (loss, per-rs RMSE).

    Per-r_s loss = main RMSE + ``tail_weight`` * tail RMSE.  The main RMSE
    uses k^2 (or uniform) weight on the full [0, kmax] grid; the tail RMSE
    uses k^p weight on k/k_F > tail_cutoff and is added in only when
    ``tail_weight > 0`` and the tail region is non-empty.

    The combined per-r_s losses are then folded into a scalar via
    ``aggregate_rmse(per_rs, mode=loss_mode)``.  ``loss_mode='sum'`` (the
    default) returns sum_i RMSE_i, which is the loss the NN is trained to
    reduce; ``'rms'`` is the legacy sqrt(mean(...^2)) form.

    Requires ``work_dir/model.json``.
    """
    run_rdmft_nk(
        exe,
        work_dir,
        rs_list,
        work_dir / "nk",
        n_grid=n_grid,
        kmax=kmax,
        init_uniform=init_uniform,
        log=log,
        label=label,
    )
    stem = NN_NK_STEM
    per_rs: dict[float, float] = {}
    prefix = f"[{label}] " if label else ""
    for rs in rs_list:
        nk_path = work_dir / "nk" / f"{stem}_rs{rs:.4f}.tsv"
        if not nk_path.is_file():
            # The C++ driver always writes n(k) now (even on non-convergence),
            # so a truly missing file means the run crashed: full penalty.
            log.info(
                f"{prefix}r_s={rs:g}: missing nk export -> penalty {UNCONV_PENALTY:g}"
            )
            per_rs[rs] = float(UNCONV_PENALTY)
            continue
        k_m, n_m, conv = load_nk_tsv(nk_path)
        x_gz, n_gz = gz_cache[rs]
        x = np.linspace(0.0, kmax, nk)
        n_ref = np.interp(x, x_gz, n_gz)
        n_mod = np.interp(x, k_m, n_m)
        w = x * x if weight_x2 else np.ones_like(x)
        diff = n_mod - n_ref
        mse = float(_trapz(w * diff * diff, x) / _trapz(w, x))
        rmse = math.sqrt(mse)
        if conv is False:
            # SCF did not converge: report a constant per-r_s penalty larger
            # than any feasible converged RMSE so the aggregate stays finite
            # and the optimizer is biased toward weight vectors that converge
            # everywhere.  The achieved (raw) RMSE is still logged below for
            # diagnostics.
            penalized = float(UNCONV_PENALTY)
            per_rs[rs] = penalized
            log.detail(
                f"{prefix}  r_s={rs:g}: RMSE={rmse:.6f} (not converged) "
                f"-> penalty {penalized:.6f}"
            )
            continue
        # Add the optional large-k tail term.  The tail uses a k^p weight
        # on k/k_F > tail_cutoff; with tail_weight=0 (default) this is a
        # no-op and the legacy main RMSE is reported.
        tail_term = 0.0
        if tail_weight > 0.0:
            tail_term = tail_rmse_vs_gz(
                k_m, n_m, gz_cache, rs,
                kmax=kmax, nk=nk,
                tail_cutoff=tail_cutoff,
                tail_power=tail_power,
            )
        per_rs[rs] = rmse + tail_weight * tail_term
        conv_s = "ok" if conv is True else ("?" if conv is None else "no")
        if tail_weight > 0.0:
            log.detail(
                f"{prefix}  r_s={rs:g}: main={rmse:.6f}  tail(k>{tail_cutoff:g},p={tail_power:g})={tail_term:.6f}"
                f"  total={per_rs[rs]:.6f}  converged={conv_s}"
            )
        else:
            log.detail(
                f"{prefix}  r_s={rs:g}: RMSE={rmse:.6f}  converged={conv_s}"
            )
    return aggregate_rmse(per_rs, mode=loss_mode), per_rs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=REPO_ROOT / "build" / "rdmft_heg")
    ap.add_argument("--dump-gz", type=Path, default=REPO_ROOT / "build" / "dump_gz_grid")
    ap.add_argument("--rs", default=",".join(f"{r:g}" for r in DEFAULT_RS))
    ap.add_argument("--prescreen-rs", default=",".join(f"{r:g}" for r in PRESCREEN_RS))
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "build" / "nn_data",
        help="Directory with gz_targets.npz from prepare_nn_gz_data.py",
    )
    ap.add_argument(
        "--hidden",
        default=",".join(str(h) for h in DEFAULT_HIDDEN),
        help="Hidden layer sizes (default small: 4,4)",
    )
    ap.add_argument("--N", type=int, default=401, help="k-grid points for main/final")
    ap.add_argument("--prescreen-N", type=int, default=401)
    ap.add_argument("--kmax", type=float, default=3.0)
    ap.add_argument("--gz-n", type=int, default=401, help="GZ reference grid points")
    ap.add_argument("--method", default="Powell", help="SciPy minimize method")
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "build" / "nn_best")
    ap.add_argument("--init", choices=("power", "mueller", "random"), default="power")
    ap.add_argument(
        "--kernel-type",
        choices=("separable", "pair"),
        default="separable",
        help="separable: K(n_i,n_j)=f(n_i)f(n_j) with f from a 1-input MLP. "
             "pair: K(n_i,n_j)=sqrt(n_i n_j)*softplus(MLP([n_i+n_j, n_i n_j])) "
             "(non-separable, 2-input).",
    )
    ap.add_argument("--init-uniform", type=float, default=0.5)
    ap.add_argument(
        "--loss",
        choices=("sum", "rms"),
        default="sum",
        help="Aggregation mode for per-r_s RMSEs.  'sum': L = sum_i RMSE_i "
             "(the NN is trained to reduce the total RMSE across the r_s "
             "sweep against the GZ targets).  'rms': legacy sqrt(mean RMSE^2).",
    )
    ap.add_argument(
        "--tail-weight",
        type=float,
        default=0.5,
        help="Coefficient for the large-k tail RMSE term added to each "
             "per-r_s loss.  Set to 0 to disable.  Default 0.5 enforces the "
             "n(k) ~ C/k^p tail asymptote alongside the main k^2-weighted RMSE.",
    )
    ap.add_argument(
        "--tail-power",
        type=float,
        default=4.0,
        help="Exponent p for the k^p weight in the tail RMSE term "
             "(physical UEG tail constraint n(k) ~ 1/k^p).",
    )
    ap.add_argument(
        "--tail-cutoff",
        type=float,
        default=1.5,
        help="Tail region begins at k/k_F > tail_cutoff (in the same units "
             "as the loaded n(k) TSVs, which are normalised by k_F).",
    )
    ap.add_argument("--no-prescreen", action="store_true")
    ap.add_argument("--uniform-weight", action="store_true", help="Use uniform k weight (default x^2)")
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress (errors still print).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Also print rdmft_heg stdout and full shell commands.",
    )
    args = ap.parse_args()

    rs_main = [float(x) for x in args.rs.split(",") if x.strip()]
    rs_pre = [float(x) for x in args.prescreen_rs.split(",") if x.strip()]
    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]

    if not args.exe.is_file():
        raise SystemExit(f"Build the driver first: {args.exe}")
    if not args.dump_gz.is_file():
        raise SystemExit(f"Build dump_gz_grid first: {args.dump_gz}")

    log = Log(quiet=args.quiet, verbose=args.verbose)
    weight_x2 = not args.uniform_weight
    in0 = 1 if args.kernel_type == "separable" else 2
    sizes = [in0] + hidden + [1]
    n_params = sum(
        sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1)
    ) + 1

    log.info("=" * 60)
    log.info(f"NN {args.kernel_type} kernel optimization vs GZ n(k)")
    log.info("=" * 60)
    log.info(f"  exe       : {args.exe.resolve()}")
    log.info(f"  data_dir  : {args.data_dir.resolve()}")
    log.info(f"  dump_gz   : {args.dump_gz.resolve()}")
    log.info(f"  out_dir   : {args.out_dir.resolve()}")
    log.info(f"  kernel    : {args.kernel_type} (input dim {in0})")
    log.info(f"  hidden    : {hidden}  ({n_params} parameters)")
    log.info(f"  method    : {args.method}  maxiter={args.maxiter}")
    log.info(f"  weight    : {'k^2' if weight_x2 else 'uniform'}")
    log.info(f"  loss      : {args.loss}  (per-r_s = main_RMSE + {args.tail_weight:g}*tail_RMSE)")
    if args.tail_weight > 0.0:
        log.info(
            f"  tail      : k/k_F > {args.tail_cutoff:g}  weight=k^{args.tail_power:g}  "
            "(enforces n(k) ~ C/k^p large-k asymptote)"
        )
    log.info(f"  init      : {args.init}  init_uniform={args.init_uniform}")
    log.info(f"  main r_s  : {rs_main}")
    log.info(f"  main grid : N={args.N}  kmax={args.kmax}")
    log.info(f"  prescreen : r_s={rs_pre}  N={args.prescreen_N}  (skip with --no-prescreen)")
    log.info("")

    gz_main = ensure_gz_targets(
        args.data_dir,
        args.dump_gz,
        rs_main,
        args.gz_n,
        args.kmax,
        weight_x2,
        log,
    )
    gz_pre = {rs: gz_main[rs] for rs in rs_pre if rs in gz_main}

    sweep_alpha = load_power_sweep_init_alpha(args.data_dir)
    if sweep_alpha is not None:
        log.info(f"  power_sweep best alpha from cache: {sweep_alpha:g}")

    candidates: list[tuple[str, dict]] = []
    if args.kernel_type == "pair":
        # Pair-mode warm-starts mirror the separable family but evaluate
        # K(a, b) = (a*b)^alpha at the chosen alpha; the trainer is then free
        # to deform the kernel into a non-factorizable shape.
        if args.init == "power":
            a0 = sweep_alpha if sweep_alpha is not None else 0.55
            candidates.append(
                (f"pair_power{a0:g}", init_pair_model("power", hidden, a0))
            )
            if sweep_alpha is None or abs(a0 - 0.55) > 1e-6:
                candidates.append(
                    ("pair_power055", init_pair_model("power", hidden, 0.55))
                )
        elif args.init == "mueller":
            candidates.append(("pair_mueller", init_pair_model("mueller", hidden)))
        else:
            candidates.append(("pair_random", build_model(hidden, kernel_type="pair")))
        # HF-like baseline (alpha = 1) so the optimizer always has at least
        # one converged-everywhere starting point on the standard r_s grid.
        candidates.append(("pair_hf", init_pair_model("hf", hidden)))
        candidates.append(("pair_power058", init_pair_model("power", hidden, 0.58)))
    else:
        if args.init == "power":
            a0 = sweep_alpha if sweep_alpha is not None else 0.55
            candidates.append((f"power{a0:g}", init_power_model(a0, hidden)))
            if sweep_alpha is None or abs(a0 - 0.55) > 1e-6:
                candidates.append(("power055", init_power_model(0.55, hidden)))
        elif args.init == "mueller":
            candidates.append(("mueller", init_power_model(0.5, hidden)))
        else:
            candidates.append(("random", build_model(hidden)))
        candidates.append(("power058", init_power_model(0.58, hidden)))

    best_model = candidates[0][1]
    best_rmse = float("inf")

    if not args.no_prescreen:
        log.info("-" * 60)
        log.info("Prescreen inits on coarse grid")
        log.info("-" * 60)
        for label, model in candidates:
            log.info(f"\n>>> prescreen candidate: {label}")
            with tempfile.TemporaryDirectory(prefix="nn_gz_pre_") as tmp:
                tmp_path = Path(tmp)
                model_path = tmp_path / "model.json"
                write_model_json(model_path, model)
                t_cand = time.time()
                try:
                    rmse, per = gz_rmse(
                        args.exe,
                        args.dump_gz,
                        rs_pre,
                        gz_pre,
                        n_grid=args.prescreen_N,
                        kmax=args.kmax,
                        nk=args.gz_n,
                        weight_x2=weight_x2,
                        init_uniform=args.init_uniform,
                        work_dir=tmp_path,
                        log=log,
                        label=label,
                        loss_mode=args.loss,
                        tail_weight=args.tail_weight,
                        tail_power=args.tail_power,
                        tail_cutoff=args.tail_cutoff,
                    )
                except (RuntimeError, FileNotFoundError) as e:
                    log.info(f"  {label}: FAILED ({time.time() - t_cand:.1f}s): {e}")
                    continue
                log.info(
                    f"  {label}: loss={rmse:.6f} ({args.loss}) ({time.time() - t_cand:.1f}s)"
                )
                log.detail(f"    {format_per_rs(per)}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    log.info(f"  ** new prescreen best ({label}) **")
        log.info(f"\nPrescreen winner: RMSE={best_rmse:.6f}")
    else:
        log.info("Skipping prescreen (--no-prescreen)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.json"
    write_model_json(model_path, best_model)
    theta0 = pack_params(best_model)
    n_eval = [0]
    best_rmse_main = [float("inf")]
    t_opt_start = time.time()

    def objective(theta: np.ndarray) -> float:
        n_eval[0] += 1
        m = unpack_params(best_model, theta)
        write_model_json(model_path, m)
        log.info(f"\n--- objective eval #{n_eval[0]} ---")
        log.detail(
            f"  |theta|={np.linalg.norm(theta):.4g}  "
            f"min={theta.min():.4g}  max={theta.max():.4g}"
        )
        with tempfile.TemporaryDirectory(prefix="nn_gz_eval_") as tmp:
            tmp_path = Path(tmp)
            write_model_json(tmp_path / "model.json", m)
            t0 = time.time()
            try:
                rmse, per = gz_rmse(
                    args.exe,
                    args.dump_gz,
                    rs_main,
                    gz_main,
                    n_grid=args.N,
                    kmax=args.kmax,
                    nk=args.gz_n,
                    weight_x2=weight_x2,
                    init_uniform=args.init_uniform,
                    work_dir=tmp_path,
                    log=log,
                    label=f"eval{n_eval[0]}",
                    loss_mode=args.loss,
                    tail_weight=args.tail_weight,
                    tail_power=args.tail_power,
                    tail_cutoff=args.tail_cutoff,
                )
            except (RuntimeError, FileNotFoundError) as e:
                rmse = float("inf")
                per = {}
                log.info(f"  eval failed: {e}")
            dt = time.time() - t0
            improved = ""
            if rmse < best_rmse_main[0]:
                best_rmse_main[0] = rmse
                improved = "  ** new best **"
            log.info(f"  loss={rmse:.6f} ({args.loss})  ({dt:.1f}s){improved}")
            log.detail(f"    {format_per_rs(per)}")
            return rmse

    def scipy_callback(xk: np.ndarray) -> None:
        log.info(
            f"  [scipy callback] iter checkpoint  |theta|={np.linalg.norm(xk):.4g}"
        )

    log.info("")
    log.info("-" * 60)
    log.info(f"Main optimization ({args.method}, maxiter={args.maxiter})")
    log.info("-" * 60)
    log.info(f"  starting loss target from prescreen: {best_rmse:.6f}")
    minimize_opts: dict[str, Any] = {"maxiter": args.maxiter}
    if not args.quiet:
        minimize_opts["disp"] = True
    res = sco.minimize(
        objective,
        theta0,
        method=args.method,
        options=minimize_opts,
        callback=scipy_callback if not args.quiet else None,
    )
    t_opt = time.time() - t_opt_start
    final_model = unpack_params(best_model, res.x)
    write_model_json(model_path, final_model)

    log.info("")
    log.info("=" * 60)
    log.info("Optimization finished")
    log.info("=" * 60)
    log.info(f"  wall time     : {t_opt:.1f}s  ({n_eval[0]} objective evals)")
    log.info(f"  scipy success : {res.success}")
    log.info(f"  scipy message : {res.message}")
    log.info(f"  nfev          : {res.nfev}")
    log.info(f"  final loss    : {res.fun:.6f}  (mode={args.loss}, tail_w={args.tail_weight:g})")
    log.info(f"  model written : {model_path}")

    # Per-r_s breakdown on final parameters
    log.info("\nFinal per-r_s RMSE (full grid)...")
    final_rmse, final_per = gz_rmse(
        args.exe,
        args.dump_gz,
        rs_main,
        gz_main,
        n_grid=args.N,
        kmax=args.kmax,
        nk=args.gz_n,
        weight_x2=weight_x2,
        init_uniform=args.init_uniform,
        work_dir=args.out_dir,
        log=log,
        label="final",
        loss_mode=args.loss,
        tail_weight=args.tail_weight,
        tail_power=args.tail_power,
        tail_cutoff=args.tail_cutoff,
    )
    log.info(f"  aggregate loss={final_rmse:.6f}  (mode={args.loss})")
    log.detail(f"    {format_per_rs(final_per)}")

    nk_dir = args.out_dir / "nk"
    log.info("\nExporting n(k) TSVs for plotting...")
    run_rdmft_nk(
        args.exe,
        args.out_dir,
        rs_main,
        nk_dir,
        n_grid=args.N,
        kmax=args.kmax,
        init_uniform=args.init_uniform,
        log=log,
        label="export",
    )
    log_path = args.out_dir / "optimize.log"
    with log_path.open("w") as f:
        f.write(f"method={args.method}\n")
        f.write(f"rmse={res.fun}\n")
        f.write(f"final_rmse={final_rmse}\n")
        f.write(f"per_rs={final_per}\n")
        f.write(f"success={res.success}\n")
        f.write(f"message={res.message}\n")
        f.write(f"nfev={res.nfev}\n")
        f.write(f"n_objective_evals={n_eval[0]}\n")
        f.write(f"wall_s={t_opt:.1f}\n")
        f.write(f"hidden={hidden}\n")
        f.write(f"rs={rs_main}\n")
    log.info(f"Log: {log_path}")
    log.info(f"n(k) TSVs: {nk_dir}")


if __name__ == "__main__":
    main()
