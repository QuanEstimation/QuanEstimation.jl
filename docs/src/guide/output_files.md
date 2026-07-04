# **Output files**

In QuanEstimation, the output data will be saved into files during or after the optimization
process. There are two categories: the values of the objective function at each step, and the
optimized variables from the corresponding scheme. This guide describes all output file types
and how to load them.

## **Objective function values**

During optimization, the objective function is evaluated at each step and the values are saved
sequentially (one value per line) into `f.csv`.

```jl
using DelimitedFiles
f = readdlm("f.csv")
```

## **Control optimization**

The control optimization results are saved into `controls.dat` (JLD2 format). The shape
depends on `savefile`:

- `savefile=false` (default): the final control coefficients are saved. The data is a vector
  of vectors, where the outer length is the number of control Hamiltonians and each inner
  vector has length `n_tseg` (the number of time segments).
- `savefile=true`: all control coefficients after each round of optimization are saved
  as a vector of vectors of vectors, with shape `(n_rounds, n_ctrl, n_tseg)`.

```jl
using JLD2
jldopen("controls.dat", "r") do f
    controls = f["controls"]
end
```

See also: [Control optimization](guide_Copt.md)

## **State optimization**

The state optimization results are saved into `states.dat` (JLD2 format).

- `savefile=false` (default): the final optimized state vectors are saved.
- `savefile=true`: all state vectors after each round are saved.

```jl
using JLD2
jldopen("states.dat", "r") do f
    states = f["states"]
end
```

See also: [State optimization](guide_Sopt.md)

## **Measurement optimization**

The measurement optimization results are saved into `measurements.dat` (JLD2 format).

- `savefile=false` (default): the final optimized POVMs are saved.
- `savefile=true`: all POVM lists after each round are saved.

```jl
using JLD2
jldopen("measurements.dat", "r") do f
    M = f["measurements"]
end
```

See also: [Measurement optimization](guide_Mopt.md)

## **Comprehensive optimization**

Comprehensive optimization combines multiple variable types. The output follows the same
conventions as above:

- Controls are saved in `controls.dat`
- States are saved in `states.dat`
- Measurements are saved in `measurements.dat`

Each file is present only if that variable type was part of the optimization. For example,
SC optimization produces `controls.dat` and `states.dat`, but not `measurements.dat`.

See also: [Comprehensive optimization](guide_Compopt.md)

## **Bayesian estimation**

The `Bayes()` and `MLE()` estimators produce JLD2 files `bayes.dat` and `MLE.dat`
respectively. Both are always saved regardless of the `savefile` setting.

| Function | File | Keys | Contents |
|----------|------|------|----------|
| `Bayes()` | `bayes.dat` | `"p"`, `"x"` | Posterior distribution(s) and estimated values |
| `MLE()` | `MLE.dat` | `"L"`, `"x"` | Likelihood function(s) and estimated values |

- `savefile=false` (default): the file contains a single posterior/likelihood and the full
  vector of estimates. `"p"` is wrapped as `[p]` (length-1 vector).
- `savefile=true`: the file contains all posteriors/likelihoods at each iteration.

```jl
using JLD2

# Bayes estimation
jldopen("bayes.dat", "r") do f
    p = f["p"]
    x = f["x"]
end

# MLE estimation
jldopen("MLE.dat", "r") do f
    L = f["L"]
    x = f["x"]
end
```

See also: [Quantum metrological tools](guide_bounds.md)

## **Adaptive measurement schemes**

Adaptive estimation via `adapt!()` produces `adaptive.dat` (JLD2 format), containing:

| Key | Contents |
|-----|----------|
| `"p"` | Posterior distributions at each adaptive step |
| `"x"` | Estimated values at each adaptive step |
| `"y"` | Experimental outcomes at each step |

```jl
using JLD2
jldopen("adaptive.dat", "r") do f
    p = f["p"]
    x = f["x"]
    y = f["y"]
end
```

See also: [Adaptive measurement schemes](guide_adaptive.md)
