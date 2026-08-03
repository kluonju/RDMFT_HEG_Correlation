#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run(binary: Path, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    require(len(sys.argv) == 2, "usage: test_cli.py <rdmft_heg>")
    binary = Path(sys.argv[1]).resolve()
    require(binary.is_file(), f"binary not found: {binary}")

    with tempfile.TemporaryDirectory(prefix="rdmft_cli_test_") as tmp:
        work = Path(tmp)
        invalid_cases = [
            ("missing option value", ["--funcs"]),
            ("option consumed as value", ["--funcs", "--force"]),
            ("non-finite rs", ["--funcs", "HF", "--rs", "nan"]),
            ("non-positive rs", ["--funcs", "HF", "--rs", "0"]),
            ("even grid", ["--funcs", "HF", "--N", "10"]),
            ("small kmax", ["--funcs", "HF", "--kmax", "1"]),
            (
                "occupation outside bounds",
                ["--funcs", "HF", "--init-uniform", "1.1"],
            ),
            ("malformed Power", ["--funcs", "Power@bad"]),
            ("out-of-range Power", ["--funcs", "Power@0"]),
            ("out-of-range Beta", ["--funcs", "Beta@1.1"]),
            ("malformed OptGeo", ["--funcs", "OptGeo@1;2junk"]),
            ("out-of-range HybOpt", ["--funcs", "HybOpt@0.5;1"]),
            ("unknown functional", ["--funcs", "NotAFunctional"]),
        ]
        for label, arguments in invalid_cases:
            result = run(binary, work, *arguments)
            require(result.returncode != 0, f"{label} was unexpectedly accepted")
            require("rdmft_heg:" in result.stderr, f"{label} had no diagnostic")

        malformed_model = work / "bad_model.json"
        malformed_model.write_text(
            '{"hidden":[],"kernels":[{"in":1,"out":1,'
            '"W":[[1e9999]],"b":[0]}],"out_bias":0}\n',
            encoding="utf-8",
        )
        result = run(binary, work, "--funcs", f"NN@{malformed_model}")
        require(result.returncode != 0, "non-finite NN weights were accepted")

        output = work / "output"
        result = run(
            binary,
            work,
            "--funcs",
            "HF",
            "--rs",
            "2",
            "--N",
            "31",
            "--out-dir",
            str(output),
            "--force",
        )
        require(result.returncode == 0, f"valid CLI failed:\n{result.stderr}")
        energy_file = output / "HF.tsv"
        require(energy_file.is_file(), "valid run did not publish HF.tsv")
        require(
            energy_file.read_text(encoding="utf-8").startswith("# rs\tfunctional"),
            "published energy TSV is incomplete",
        )
        require(not list(output.glob("*.tmp.*")), "successful run left temporary files")

        energy_file.write_text("sentinel\n", encoding="utf-8")
        result = run(
            binary,
            work,
            "--funcs",
            "HF",
            "--rs",
            "1e-200",
            "--N",
            "3",
            "--out-dir",
            str(output),
            "--force",
        )
        require(result.returncode != 0, "overflowing run unexpectedly succeeded")
        require(
            energy_file.read_text(encoding="utf-8") == "sentinel\n",
            "failed run replaced the previous complete TSV",
        )
        require(not list(output.glob("*.tmp.*")), "failed run left temporary files")

    print("test_cli: all checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
