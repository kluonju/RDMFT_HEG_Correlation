#!/usr/bin/env python3
"""Generate tests/fixtures/nn_power055.json (stdlib only)."""
import json
import math
import random
from pathlib import Path

random.seed(42)


def build(hidden):
    sizes = [1] + hidden + [1]
    kernels = []
    for ell in range(len(sizes) - 1):
        in_d, out_d = sizes[ell], sizes[ell + 1]
        W = [[random.gauss(0, 0.05) for _ in range(in_d)] for _ in range(out_d)]
        b = [0.0] * out_d
        kernels.append({"in": in_d, "out": out_d, "W": W, "b": b})
    anchors = [0.05, 0.15, 0.35, 0.55, 0.75, 0.95]
    alpha = 0.55
    raw = []
    for a in anchors:
        t = a**alpha / a
        raw.append(math.log(math.expm1(max(t, 1e-8))))
    out_bias = sum(raw) / len(raw)
    return {
        "version": 1,
        "name": "NN_power055_init",
        "hidden": hidden,
        "kernels": kernels,
        "out_bias": out_bias,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("nn_power055.json")
    out.write_text(json.dumps(build([8, 8]), indent=2))
    print("wrote", out)
