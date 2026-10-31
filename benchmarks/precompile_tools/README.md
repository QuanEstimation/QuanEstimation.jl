# Public 0.5: PrecompileTools first-call benchmark

This directory contains the reproducible benchmark for the **public 0.5 only**
backport of the 1.0 PrecompileTools work. It does not modify or measure the
internal 1.0 development tree.

## Configurations

The requested end-to-end comparison is:

| Label | Source | Julia | Precompile workload |
|---|---|---|---|
| `public05-julia1.12-baseline` | public 0.5 commit `f949915` | 1.12.7 | absent |
| `public05-julia1.13-precompiletools` | the same commit plus this backport | 1.13.1 | Base + Root |

This is intentionally a **combined runtime + patch comparison**. It must not be
used to attribute the entire improvement to PrecompileTools alone. The 1.0
historical measurements use a different API and are not baseline values for
this table.

Twelve paths are baked. The CFIM measurement-optimization scenario is retained
in the benchmark but deliberately excluded from `@compile_workload`: baking
its Zygote pullback caused a reproducible multiparameter `StateOpt`/`CFIM`
regression (`Vector{Any} is not callable`). Removing only that cell restored
the path; autoGRAPE remains baked. This negative result is part of the 0.5
backport boundary, not a failure hidden by the timing table.

Both legs use the same machine, public API fixtures and benchmark script, one
Julia thread, one OpenBLAS thread and `/usr/sbin/python3` through
`JULIA_PYTHONCALL_EXE`. Fixing PythonCall's interpreter prevents CondaPkg
resolution from becoming an unrelated benchmark variable; none of the measured
numerical scenarios calls Python.

## Measurement protocol

1. Create a new depot with an empty `compiled/` directory. Package sources,
   artifacts and registries are symlinked from the normal depot, so downloads
   are excluded but no existing compiled cache is reused.
2. Resolve/instantiate before timing. Record the exact Julia version, Git HEAD,
   a SHA-256 fingerprint of tracked changes and untracked source files, and the
   resolved Manifest hash.
3. Run `Pkg.precompile("QuanEstimation"; timing=true)` and record total cold
   wall time, the individual Base/NV/Root timings and package-image sizes.
4. Start a fresh Julia process for each scenario. Measure `using
   QuanEstimation`, then the first public-API call, the second call and five
   further warm calls. Report the median of those five warm calls.
5. Save raw logs and `results.json`. First/second/warm records include elapsed
   time, JIT time, recompilation time, GC time, allocated bytes and a numerical
   checksum.

The optimization workloads use temporary directories because legacy 0.5
`optimize!` and online MZI write files even with `savefile=false`. MZI metrics
include reading the actual estimator/outcome record from `adaptive.dat`;
returning the input schedule would not validate the calculation. Measurement
optimization returns the optimized `opt.B`, not the CFIM of the untouched
default Scheme measurement.

## Reproduce

From `public/QuanEstimation.jl`:

```bash
python3 benchmarks/precompile_tools/run_benchmarks.py \
  --source /path/to/public-05-baseline \
  --channel 1.12 --expect-version 1.12.7 \
  --label public05-julia1.12-baseline \
  --output /path/to/results/public05-julia1.12-baseline

python3 benchmarks/precompile_tools/run_benchmarks.py \
  --source /path/to/public-05-with-precompiletools \
  --channel 1.13 --expect-version 1.13.1 \
  --label public05-julia1.13-precompiletools \
  --output /path/to/results/public05-julia1.13-precompiletools
```

The baseline source can be a detached worktree at `f949915`; copy this
`benchmarks/precompile_tools` directory there without applying the package
changes. Run the two legs sequentially, not concurrently.

## Results (2026-10-29)

Complete records:

- [All first-call, warm and compilation metrics](results/summary-20261029.md)
- [1.12.7 baseline JSON](results/public05-julia1.12-baseline-20261029.json)
- [1.13.1 + PrecompileTools JSON](results/public05-julia1.13-precompiletools-20261029.json)
- [Final-run raw logs](results/logs/)

The measurements ran sequentially on an Intel Xeon E-2236 (10 logical CPUs,
31 GiB RAM), with one Julia/OpenBLAS thread. The baseline source fingerprint
was `3d30e47206181822bda242145d1d7290dc7aaaba135fd933f5b115e82a7d970e`;
the final modified source fingerprint was
`660b13d52fc713806ad0f1fba8a2244c3484e51861180b7d4a78c0bccd9634f7`.

### Representative public API results

Warm is the median of calls 3–7. See the complete summary for all 13 paths and
for JIT/recompilation details.

| Scenario | 1.12.7 baseline first | 1.13.1 + PCT first | Speedup | Baseline warm | Improved warm |
|---|---:|---:|---:|---:|---:|
| QFIM | 9.860 s | 58.28 ms | 169.2× | 128.9 µs | 127.4 µs |
| GRAPE | 19.161 s | 259.67 ms | 73.8× | 13.54 ms | 13.22 ms |
| autoGRAPE | 52.372 s | 225.77 ms | 232.0× | 15.41 ms | 15.11 ms |
| ODE control optimization | 21.966 s | 472.71 ms | 46.5× | 61.21 ms | 63.75 ms |
| MZI online + result read | 8.012 s | 476.43 ms | 16.8× | 268.88 ms | 263.55 ms |
| Measurement optimization (not baked) | 70.884 s | 34.929 s | 2.0× | 15.51 ms | 15.38 ms |
| NHB / SDP | 22.944 s | 280.06 ms | 81.9× | 9.63 ms | 10.61 ms |

All 13 numerical checksums agree across the configurations at relative
tolerance `1e-8` / absolute tolerance `1e-10`. The 12 baked paths reported zero
`recompile_time`; the deliberately unbaked measurement optimization retained
0.184 s of recompilation time. Warm timings remain in the same operational
range, but these single-run microbenchmarks do not establish a universal warm
speedup: small regressions and improvements are both retained in the table.

### Precompile, image and load cost

| Metric | 1.12.7 baseline | 1.13.1 + PrecompileTools |
|---|---:|---:|
| Full cold precompile wall | 257.432 s | 289.216 s |
| QuanEstimationBase precompile | 11.356 s | 89.700 s |
| NVMagnetometer precompile | 8.858 s | 6.900 s |
| QuanEstimation Root precompile | 7.016 s | 7.400 s |
| Base `.ji + .so` | 4.83 MiB | 88.03 MiB |
| NV `.ji + .so` | 0.16 MiB | 0.18 MiB |
| Root `.ji + .so` | 0.07 MiB | 0.10 MiB |
| Median `using QuanEstimation` | 5.716 s | 5.599 s |

The cold-wall comparison includes the Julia runtime change and all dependencies;
it is not a pure PrecompileTools cost. The package-specific Base row is the
clearest workload cost indicator in this protocol. No partial or failed run is
included in these tables.

## Validation

- Julia 1.12.7, modified 0.5: Base/Root precompilation succeeded and all 37
  workload semantic tests passed. This compatibility run is separate from the
  formal benchmark table.
- Julia 1.13.1, `--check-bounds=yes`: full `Pkg.test()` passed **1437/1437**
  tests in 6 min 21.5 s (the historical 1400 tests plus 37 workload tests).
- During development, baking the CFIM measurement AD cell produced one
  multiparameter StateOpt/CFIM error. The unmodified baseline passed the
  isolated path; removing only that baked cell restored it and then produced
  the 1437/1437 result above.
