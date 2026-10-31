# Warm-kernel measurement shared by the baseline and modified 0.5 trees.
# Usage: julia --project=. benchmarks/numerical_backports/measure.jl <case>

length(ARGS) == 1 || error("usage: measure.jl <case>")
using QuanEstimation
using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "fixtures.jl"))
const CASE_NAME = ARGS[1]
haskey(NUMERICAL_CASES, CASE_NAME) || error("unknown case: $CASE_NAME")
const case = NUMERICAL_CASES[CASE_NAME]

allocations(stats) = begin
    gc = stats.gcstats
    gc.malloc + gc.realloc + gc.poolalloc + gc.bigalloc
end
median(values) = begin
    ordered = sort(collect(values))
    n = length(ordered)
    isodd(n) ? ordered[(n + 1) ÷ 2] : (ordered[n ÷ 2] + ordered[n ÷ 2 + 1]) / 2
end

# Two untimed warmups exclude method/JIT costs from this numerical-kernel A/B.
result = case()
finite_numerical_result(result) || error("non-finite warm result: $CASE_NAME")
result = case()
finite_numerical_result(result) || error("non-finite warm result: $CASE_NAME")
samples = [@timed case() for _ in 1:7]
for sample in samples
    finite_numerical_result(sample.value) || error("non-finite result: $CASE_NAME")
end
result = samples[end].value

if haskey(ENV, "NUMERICAL_RESULT_FILE")
    path = ENV["NUMERICAL_RESULT_FILE"]
    mkpath(dirname(path))
    QuanEstimationBase.jldopen(path, "w") do file
        for (key, value) in numerical_result_payload(result)
            file[key] = value
        end
    end
end

println(
    "METRIC case=", CASE_NAME,
    " julia=", VERSION,
    " time_min_s=", minimum(sample.time for sample in samples),
    " time_median_s=", median(sample.time for sample in samples),
    " time_max_s=", maximum(sample.time for sample in samples),
    " allocations_median=", median(allocations(sample) for sample in samples),
    " bytes_median=", median(sample.bytes for sample in samples),
    " gc_median_s=", median(sample.gctime for sample in samples),
    " checksum=", numerical_checksum(result),
)
