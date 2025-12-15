using QuanEstimation
using Random
using Plots
using LinearAlgebra

# free Hamiltonian
function H0_func(x)
    return 0.5*B*omega0*(sx*cos(x)+sz*sin(x))
end
# derivative of the free Hamiltonian on x
function dH_func(x)
    return [0.5*B*omega0*(-sx*sin(x)+sz*cos(x))]
end

B, omega0 = pi/2.0, 1.0
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
# initial state
rho0 = 0.5*ones(2, 2)
# measurement 
M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
M = [M1, M2]
# prior distribution
x = range(0., stop=0.5*pi, length=1000) |>Vector
p = (1.0/(x[end]-x[1]))*ones(length(x))
# time length for the evolution
tspan = range(0., stop=1., length=1000)
# dynamics
rho = Vector{Matrix{ComplexF64}}(undef, length(x))
for i in 1:length(x) 
    H0_tp = H0_func(x[i])
    dH_tp = dH_func(x[i])
    rho_tp, drho_tp = QuanEstimation.expm(tspan, rho0, H0_tp, dH_tp)
    rho[i] = rho_tp[end]
end

x_real = 0.2*pi
measurement_counts = [500, 2500, 5000]
colors = [:blue, :red, :green]

p_combined = plot()

map_estimates = Float64[]
mle_estimates = Float64[]
pout_results = []

for (idx, N) in enumerate(measurement_counts)
    println("Processing  measurements...")
    
    # Use different seeds for reproducibility across N
    Random.seed!(idx)
    
    local y = []
    local H0_real = H0_func(x_real)
    local dH_real = dH_func(x_real)
    local rho_real, drho_real = QuanEstimation.expm(tspan, rho0, H0_real, dH_real)
    local rho_real = rho_real[end]
    local p1 = real(tr(M[1]*rho_real))
    
    for i in 1:N
        if rand() < p1
            push!(y, 0)
        else
            push!(y, 1)
        end
    end
    
    #===============Maximum a posteriori estimation===============#
    pout, xout = QuanEstimation.Bayes([x], p, rho, y; M=M, estimator="MAP", savefile=false)

    #===============Maximum likelihood estimation===============#
    Lout, xout_mle = QuanEstimation.MLE([x], rho, y, M=M; savefile=false)
    
    local pout_normalized = pout ./ sum(pout)
    local dx = x[2] - x[1]
    local pout_density = pout_normalized ./ dx

    push!(map_estimates, xout)
    push!(mle_estimates, xout_mle)
    push!(pout_results, pout_density)
    
    # Plot posterior
    plot!(p_combined, x, pout_density, 
          linewidth=2,
          color=colors[idx],
          label=measurement_counts[idx],
          alpha=0.8)

end

# Add a single vertical line marking the true value
vline!(p_combined, [x_real];
       linewidth = 3,
       linestyle = :dash,
       alpha = 0.9,
       label = "true x")

# Zoom x-limits around posterior peaks (optional)
peak_positions = [x[argmax(pout)] for pout in pout_results]
x_min = max(minimum(peak_positions) - 0.1, x[1])
x_max = min(maximum(peak_positions) + 0.1, x[end])
xlims!(p_combined, (x_min, x_max))

plot!(p_combined,
      title="Posterior Probability Distribution Comparison",
      xlabel="Parameter x (radians)", 
      ylabel="Posterior Probability P(x|y)",
      legend=:topright,
      grid=true,
      framestyle=:box)

display(p_combined)
