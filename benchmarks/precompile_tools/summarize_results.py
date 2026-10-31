#!/usr/bin/env python3
"""Render two run_benchmarks.py results.json files as Markdown tables."""

import argparse
import json
import math
from pathlib import Path
import statistics


def seconds(value):
    if value < 0.001:
        return f"{value * 1e6:.1f} µs"
    if value < 1:
        return f"{value * 1000:.2f} ms"
    return f"{value:.3f} s"


def mib(value):
    return value / (1024**2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("improved", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    improved = json.loads(args.improved.read_text(encoding="utf-8"))

    lines = [
        "### Protocol and precompile cost",
        "",
        "| Metric | 1.12.7 baseline | 1.13.1 + PrecompileTools |",
        "|---|---:|---:|",
        f"| Cold precompile wall | {seconds(baseline['cold_precompile_wall_s'])} | {seconds(improved['cold_precompile_wall_s'])} |",
    ]
    for package in ("QuanEstimationBase", "NVMagnetometer", "QuanEstimation"):
        lines.append(
            f"| {package} precompile | {seconds(baseline['package_precompile_s'][package])} "
            f"| {seconds(improved['package_precompile_s'][package])} |"
        )
        lines.append(
            f"| {package} images (.ji + .so) | {mib(baseline['package_images'][package]['total_bytes']):.2f} MiB "
            f"| {mib(improved['package_images'][package]['total_bytes']):.2f} MiB |"
        )
    baseline_load = statistics.median(
        item["metrics"]["first"]["load_s"] for item in baseline["scenarios"].values()
    )
    improved_load = statistics.median(
        item["metrics"]["first"]["load_s"] for item in improved["scenarios"].values()
    )
    lines.append(f"| Median `using QuanEstimation` | {seconds(baseline_load)} | {seconds(improved_load)} |")

    lines.extend(
        [
            "",
            "### Public API first-call and warm performance",
            "",
            "Warm is the median of calls 3–7 in the same fresh process. The checksum column compares numerical results across the two legs (relative tolerance 1e-8, absolute tolerance 1e-10).",
            "",
            "| Scenario | Baseline first | Improved first | First speedup | Baseline warm | Improved warm | Checksum |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for scenario in baseline["scenarios"]:
        before = baseline["scenarios"][scenario]["metrics"]
        after = improved["scenarios"][scenario]["metrics"]
        before_first = before["first"]["time_s"]
        after_first = after["first"]["time_s"]
        same = math.isclose(
            before["first"]["checksum"],
            after["first"]["checksum"],
            rel_tol=1e-8,
            abs_tol=1e-10,
        )
        lines.append(
            f"| `{scenario}` | {seconds(before_first)} | {seconds(after_first)} "
            f"| {before_first / after_first:.1f}× | {seconds(before['warm5']['time_s'])} "
            f"| {seconds(after['warm5']['time_s'])} | {'✓' if same else '⚠'} |"
        )

    lines.extend(
        [
            "",
            "### First-call compilation detail",
            "",
            "| Scenario | Baseline JIT / recompile | Improved JIT / recompile |",
            "|---|---:|---:|",
        ]
    )
    for scenario in baseline["scenarios"]:
        before = baseline["scenarios"][scenario]["metrics"]["first"]
        after = improved["scenarios"][scenario]["metrics"]["first"]
        lines.append(
            f"| `{scenario}` | {seconds(before['compile_s'])} / {seconds(before['recompile_s'])} "
            f"| {seconds(after['compile_s'])} / {seconds(after['recompile_s'])} |"
        )

    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
