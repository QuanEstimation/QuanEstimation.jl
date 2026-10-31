#!/usr/bin/env python3
"""Summarize baseline/candidate METRIC logs for the numerical backports."""

import argparse
from pathlib import Path


CASES = ["mzi", "mzi_offline", "mzi_offline_mi", "evolve_ode", "pso_ode"]


def read_metric(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("METRIC "):
            return dict(token.split("=", 1) for token in line[7:].split())
    raise RuntimeError(f"no METRIC line in {path}")


def seconds(value):
    if value < 0.001:
        return f"{value * 1e6:.2f} µs"
    if value < 1:
        return f"{value * 1000:.3f} ms"
    return f"{value:.4f} s"


def mb(value):
    return value / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for case in CASES:
        before = read_metric(args.results / "baseline" / f"{case}.log")
        after = read_metric(args.results / "candidate" / f"{case}.log")
        if before["checksum"] != after["checksum"]:
            raise RuntimeError(f"checksum mismatch for {case}")
        rows.append((case, before, after))

    lines = [
        "# Public 0.5 numerical-backport results (2026-10-29)",
        "",
        "Both legs include the earlier PrecompileTools backport and run on Julia 1.13.1 with one Julia/OpenBLAS thread. Every case performs two warmups and seven measured calls. Time min/median are reported; allocations and bytes are medians.",
        "",
        "| Case | Time min before → after | Speedup | Time median before → after | Allocations before → after | Bytes before → after | Byte reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case, before, after in rows:
        bmin = float(before["time_min_s"])
        amin = float(after["time_min_s"])
        bmed = float(before["time_median_s"])
        amed = float(after["time_median_s"])
        balloc = float(before["allocations_median"])
        aalloc = float(after["allocations_median"])
        bbytes = float(before["bytes_median"])
        abytes = float(after["bytes_median"])
        lines.append(
            f"| `{case}` | {seconds(bmin)} → {seconds(amin)} | {bmin / amin:.2f}× "
            f"| {seconds(bmed)} → {seconds(amed)} | {int(balloc):,} → {int(aalloc):,} "
            f"| {mb(bbytes):.3f} MB → {mb(abytes):.3f} MB | {bbytes / abytes:.2f}× |"
        )
    lines.extend(
        [
            "",
            "All checksums are identical. Separate `COMPARE_ALL exact=true` checks cover the MZI phase/outcome arrays, both offline scores, and the ODE trajectory/PSO controls, not just scalar checksums.",
            "",
            "GC median before → after:",
            "",
            "| Case | GC median before | GC median after |",
            "|---|---:|---:|",
        ]
    )
    for case, before, after in rows:
        lines.append(
            f"| `{case}` | {seconds(float(before['gc_median_s']))} | {seconds(float(after['gc_median_s']))} |"
        )
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
