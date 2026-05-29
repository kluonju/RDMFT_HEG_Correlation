#!/usr/bin/env python3
"""Generate tests/fixtures/nn_pair.json for the NNPair unit test (stdlib only).

Initialises a 16x16 MLP whose ``raw`` output is ~ sqrt(n_i n_j) on a few
representative anchor points (so K(n_i, n_j) ~ n_i n_j ~ HF) but with full
sign-free flexibility — the optimizer is free to deform the network into any
non-factorisable shape that better matches the GZ targets.
"""
import json
import math
import random
from pathlib import Path

random.seed(7)


def build(hidden):
    sizes = [2] + hidden + [1]
    kernels = []
    for ell in range(len(sizes) - 1):
        in_d, out_d = sizes[ell], sizes[ell + 1]
        W = [[random.gauss(0, 0.05) for _ in range(in_d)] for _ in range(out_d)]
        b = [0.0] * out_d
        kernels.append({"in": in_d, "out": out_d, "W": W, "b": b})
    # K(a,b) = sqrt(ab) * raw  ~  ab   ->   raw ~ sqrt(ab).  Fit out_bias to
    # the mean target raw at a few representative pair anchors.
    anchors = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.4, 0.7), (0.1, 0.9)]
    raw = [math.sqrt(a * b) for a, b in anchors]
    return {
        "version": 1,
        "kernel_type": "pair",
        "name": "NNPair_hf_like_init",
        "hidden": hidden,
        "kernels": kernels,
        "out_bias": sum(raw) / len(raw),
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("nn_pair.json")
    out.write_text(json.dumps(build([16, 16]), indent=2))
    print("wrote", out)
