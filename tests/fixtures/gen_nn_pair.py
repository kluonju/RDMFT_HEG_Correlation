#!/usr/bin/env python3
"""Generate tests/fixtures/nn_pair.json for the NNPair unit test (stdlib only).

Initialises a small 2-input MLP whose softplus(out) approximately matches the
HF kernel K(a, b) = a*b on a few anchor points, by setting the output bias so
that ``sqrt(p) * softplus(raw)`` ~= p where p = a*b.  The hidden layers start
small so the prefactor sqrt(p) dominates and the kernel is well-defined.
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
    # Anchor: with sqrt(p) * softplus(raw) ~ p   =>  softplus(raw) ~ sqrt(p).
    # Average the softplus pre-image across a few representative pair values
    # so the network starts close to a sensible baseline.
    anchors = [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.4, 0.7), (0.1, 0.9)]
    raw = []
    for a, b in anchors:
        target = math.sqrt(a * b)  # softplus(raw) target
        raw.append(math.log(math.expm1(max(target, 1e-6))))
    out_bias = sum(raw) / len(raw)
    return {
        "version": 1,
        "kernel_type": "pair",
        "name": "NNPair_hf_like_init",
        "hidden": hidden,
        "kernels": kernels,
        "out_bias": out_bias,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("nn_pair.json")
    out.write_text(json.dumps(build([8, 8]), indent=2))
    print("wrote", out)
