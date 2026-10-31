# Public 0.5 numerical optimization backports

This directory records the second optimization batch for **public 0.5 only**.
The baseline already contains the earlier Base/Root PrecompileTools backport;
these measurements therefore isolate the additional numerical-kernel changes.
The internal 1.0 tree was used only as the implementation reference and was not
modified.

## Backported techniques

### 1. MZI baked device operators and scratch kernels

Reference: 1.0 commit `20bb5d4` (B3a), adapted to the legacy 0.5 API in
`Scheme/EstimationStrategy/AdaptiveEstimation/AdptiveMZI.jl`.

- Bake `a ⊗ I`, `I ⊗ a`, their union sparse pattern and source-index maps once
  per device instead of rebuilding them for every phase/grid point.
- Refill `a_u` in place and use `mul!` for accumulated operators and likelihood
  evaluation. Preserve the original left-associative `(rho0 * A') * A` order.
- Hoist online `M1`, factorial factors and the final `M2` evaluation out of
  loops that recomputed identical values.
- Reuse the same fast path for online and offline sharpness/MI calculations.

The public `a_u` function remains as the reference implementation. Its sparse
pattern and every stored value are checked bitwise against the refill kernel
at Float64 and Float32. Online phase/outcome arrays and both offline scores are
also bitwise identical to the baseline.

### 2. ODE Hamiltonian hoisting and in-place commutators

Reference: the 1.0 `_assemble_H!` / `lindblad_rhs!` mechanism, adapted without
changing the 0.5 Tsit5 solver or its time indexing.

- Construct the complete time-sliced `Htot(H0, Hc, ctrl)` once per ODE solve,
  rather than twice per right-hand-side evaluation.
- Evaluate `-im * (H * rho - rho * H)` with call-local scratch and `mul!` in
  all four controlled ODE paths: direct density matrix, non-decay, decay and
  ket/decay evolution.
- Keep decay expressions, their summation order, tolerances, output files and
  public APIs unchanged.

The scratch commutator is bitwise identical to the original expression. Full
`rho`/`drho` trajectories and the fixed-seed PSO objective/control arrays are
also bitwise identical to the baseline.

Deliberately excluded from this batch: Liouvillian decomposition tables,
exact-Fréchet derivatives, RK45/dense-output changes, PSO loop parallelism and
PSO `c2` behavior fixes. Those require a separate numerical-semantics or
thread-safety review.

## Measurement protocol

- Same machine for both legs: Intel Xeon E-2236, 31 GiB RAM.
- Julia 1.13.1 for both legs, one Julia thread and one OpenBLAS thread.
- Baseline: the complete 0.5 tree immediately before this numerical batch,
  including PrecompileTools. Candidate: the same tree plus these two kernels.
- Two untimed warmups followed by seven measured calls per case.
- Report minimum and median elapsed time, median allocation count, allocated
  bytes and GC time. Save full JLD2 results, not only scalar checksums.
- `compare_results.jl` checks both bitwise equality and a tolerant fallback;
  all adopted cases pass the bitwise comparison.

Run a case with:

```bash
NUMERICAL_RESULT_FILE=/path/to/case.jld2 \
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
JULIA_PYTHONCALL_EXE=/usr/sbin/python3 \
julia +1.13 --startup-file=no --project=. \
  benchmarks/numerical_backports/measure.jl mzi
```

Compare two result files with:

```bash
julia +1.13 --startup-file=no --project=. \
  benchmarks/numerical_backports/compare_results.jl \
  /path/to/baseline.jld2 /path/to/candidate.jld2
```

Regenerate the table with `python3 benchmarks/numerical_backports/summarize.py
/path/to/results --output summary.md` after placing logs in `baseline/` and
`candidate/` subdirectories.

## Results

The complete machine-generated table is
[`results/summary-20261029.md`](results/summary-20261029.md); raw logs and JLD2
payloads are under `results/baseline/` and `results/candidate/`; all five
`COMPARE_ALL exact=true` verdicts are in `results/comparisons.log`.

| Case | Time min before → after | Speedup | Bytes before → after | Reduction |
|---|---:|---:|---:|---:|
| MZI online | 246.50 ms → 89.02 ms | 2.77× | 738.10 MB → 1.23 MB | 602.15× |
| MZI offline sharpness | 5.406 ms → 1.216 ms | 4.45× | 11.64 MB → 1.71 MB | 6.79× |
| MZI offline MI | 2.734 ms → 1.193 ms | 2.29× | 11.64 MB → 1.72 MB | 6.79× |
| ODE `evolve` | 7.760 ms → 0.719 ms | 10.80× | 34.31 MB → 1.93 MB | 17.82× |
| ODE PSO | 2.270 s → 0.177 s | 12.81× | 7.255 GB → 0.353 GB | 20.58× |

The MZI online allocation count falls from 2,296,032 to 1,955 and its median GC
time from 43.89 ms to zero. The ODE PSO allocation count falls from 106,571,473
to 5,020,729 and median GC time from 570.58 ms to 29.21 ms. These are 0.5
measurements, not a reuse of the 1.0 performance claims.

## Validation

- New `test_numerical_backports.jl`: 25/25 bitwise oracle checks passed.
- Existing PrecompileTools workload: 37/37 passed.
- Existing adaptive MZI integration suite: 13/13 passed.
- Julia 1.13.1, `--check-bounds=yes`: full `Pkg.test()` passed **1462/1462**
  in 6 min 1.1 s; the final log is `results/logs/julia1.13-pkg-test.log`.
- Julia 1.12.7: Base/Root precompilation succeeded; all 62 related tests
  (37 workload + 25 new bitwise gates) passed.
