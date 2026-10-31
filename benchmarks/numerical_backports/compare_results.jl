# Compare two NUMERICAL_RESULT_FILE payloads, with exact and tolerant verdicts.
length(ARGS) == 2 || error("usage: compare_results.jl <baseline.jld2> <modified.jld2>")
using QuanEstimation
using LinearAlgebra

function load_payload(path)
    return QuanEstimationBase.jldopen(path, "r") do file
        Dict{String, Any}(key => read(file, key) for key in keys(file))
    end
end

baseline = load_payload(ARGS[1])
modified = load_payload(ARGS[2])
keys(baseline) == keys(modified) || error("result keys differ")

function compare_value(a, b, path)
    if a isa Number && b isa Number
        exact = isequal(a, b)
        close = isapprox(a, b; rtol = 1e-10, atol = 1e-12)
        error = abs(a - b)
    elseif a isa AbstractArray && b isa AbstractArray
        size(a) == size(b) || error("shape mismatch at $path")
        exact = true
        close = true
        error = 0.0
        for index in eachindex(a, b)
            child_exact, child_close, child_error = compare_value(a[index], b[index], "$path[$index]")
            exact &= child_exact
            close &= child_close
            error = max(error, child_error)
        end
        return exact, close, error
    else
        exact = isequal(a, b)
        close = exact
        error = exact ? 0.0 : Inf
    end
    return exact, close, error
end

all_exact = true
all_close = true
for key in sort!(collect(keys(baseline)))
    exact, close, error = compare_value(baseline[key], modified[key], key)
    global all_exact = all_exact && exact
    global all_close = all_close && close
    println("COMPARE key=", key, " exact=", exact, " close=", close, " max_abs=", error)
end
println("COMPARE_ALL exact=", all_exact, " close=", all_close)
all_close || exit(1)
