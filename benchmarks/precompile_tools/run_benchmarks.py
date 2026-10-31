#!/usr/bin/env python3
"""Run the public-0.5 cold-precompile and fresh-process TTFX benchmark.

The generated depot shares package sources, artifacts and registries with the
normal Julia depot, but has an empty compiled/ directory. No benchmark timing
is taken from the package-manager resolve/instantiate phase.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import time


SCENARIOS = [
    "qfim",
    "cfim",
    "kraus",
    "ode",
    "grape",
    "autogrape",
    "de",
    "pso",
    "ode_opt",
    "waveform",
    "meas_opt",
    "mzi",
    "nhb",
]


def run_logged(command, *, env, cwd, log_path, timed=False):
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            env=env,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}); see {log_path}:\n"
            + shlex.join(command)
        )
    return elapsed if timed else None


def parse_metrics(log_path):
    samples = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("METRIC "):
            continue
        fields = dict(token.split("=", 1) for token in line[7:].split())
        sample = fields.pop("sample")
        for key, value in list(fields.items()):
            if key not in {"scenario", "julia"}:
                fields[key] = float(value)
        samples[sample] = fields
    if set(samples) != {"first", "second", "warm5"}:
        raise RuntimeError(f"incomplete metrics in {log_path}: {sorted(samples)}")
    return samples


def source_provenance(source):
    status_bytes = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "-z"]
    )
    tracked_diff = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--binary", "HEAD"]
    )
    untracked = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard", "-z"]
    ).split(b"\0")
    digest = hashlib.sha256()
    digest.update(status_bytes)
    digest.update(tracked_diff)
    for raw_name in sorted(name for name in untracked if name):
        path = source / os.fsdecode(raw_name)
        digest.update(raw_name + b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0" + os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(path.read_bytes())
    return {
        "git_status": [
            os.fsdecode(item) for item in status_bytes.split(b"\0") if item
        ],
        "source_fingerprint_sha256": digest.hexdigest(),
    }


def package_precompile_times(log_path):
    wanted = {"QuanEstimationBase", "NVMagnetometer", "QuanEstimation"}
    pattern = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s)\s+✓\s+(.+?)\s*$")
    timings = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match and match.group(3) in wanted:
            value = float(match.group(1))
            timings[match.group(3)] = value / 1000 if match.group(2) == "ms" else value
    missing = wanted - timings.keys()
    if missing:
        raise RuntimeError(f"missing package timings in {log_path}: {sorted(missing)}")
    return timings


def package_images(depot):
    images = {}
    for package in ("QuanEstimationBase", "NVMagnetometer", "QuanEstimation"):
        entries = []
        for path in sorted(depot.glob(f"compiled/v*/{package}/*")):
            if path.suffix.lower() not in {".ji", ".so", ".dylib", ".dll"}:
                continue
            entries.append(
                {
                    "path": str(path.relative_to(depot)),
                    "bytes": path.stat().st_size,
                }
            )
        images[package] = {
            "files": entries,
            "total_bytes": sum(entry["bytes"] for entry in entries),
        }
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--channel", required=True, help="juliaup channel, e.g. 1.12")
    parser.add_argument("--expect-version", help="assert the exact Julia patch version")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared-depot", type=Path, default=Path.home() / ".julia")
    parser.add_argument("--python", default=shutil.which("python3") or "python3")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    args = parser.parse_args()

    source = args.source.resolve()
    shared_depot = args.shared_depot.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    depot = output / "depot"
    depot.mkdir()
    for name in ("packages", "artifacts", "registries"):
        (depot / name).symlink_to(shared_depot / name, target_is_directory=True)

    env = os.environ.copy()
    env.update(
        {
            "JULIA_DEPOT_PATH": str(depot),
            "JULIA_PKG_PRECOMPILE_AUTO": "0",
            "JULIA_NUM_THREADS": "1",
            "JULIA_PYTHONCALL_EXE": args.python,
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    julia = ["julia", f"+{args.channel}", "--startup-file=no", f"--project={source}"]
    julia_version = subprocess.check_output(
        julia + ["-e", "print(VERSION)"], env=env, cwd=source, text=True
    ).strip()
    if args.expect_version and julia_version != args.expect_version:
        raise RuntimeError(f"expected Julia {args.expect_version}, got {julia_version}")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    result = {
        "label": args.label,
        "channel": args.channel,
        "julia_version": julia_version,
        "source": str(source),
        "git_revision": revision,
        **source_provenance(source),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_platform": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "julia_num_threads": 1,
        "openblas_num_threads": 1,
        "pythoncall_exe": args.python,
        "scenarios": {},
    }

    run_logged(
        julia + ["-e", "using Pkg; Pkg.resolve(); Pkg.instantiate()"],
        env=env,
        cwd=source,
        log_path=output / "setup.log",
    )
    result["resolved_manifest_sha256"] = hashlib.sha256(
        (source / "Manifest.toml").read_bytes()
    ).hexdigest()
    result["cold_precompile_wall_s"] = run_logged(
        julia + [
            "-e",
            'using Pkg; Pkg.precompile("QuanEstimation"; timing=true)',
        ],
        env=env,
        cwd=source,
        log_path=output / "precompile.log",
        timed=True,
    )
    result["package_precompile_s"] = package_precompile_times(output / "precompile.log")
    result["package_images"] = package_images(depot)

    for scenario in args.scenarios:
        log_path = output / f"{scenario}.log"
        process_wall_s = run_logged(
            julia + [str(source / "benchmarks/precompile_tools/first_call.jl"), scenario],
            env=env,
            cwd=source,
            log_path=log_path,
            timed=True,
        )
        result["scenarios"][scenario] = {
            "process_wall_s": process_wall_s,
            "metrics": parse_metrics(log_path),
        }
        first = result["scenarios"][scenario]["metrics"]["first"]
        print(
            f"{args.label} {scenario}: first={first['time_s']:.6g}s "
            f"second={result['scenarios'][scenario]['metrics']['second']['time_s']:.6g}s "
            f"load={first['load_s']:.6g}s",
            flush=True,
        )
        (output / "results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
