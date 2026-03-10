using MvFInfectionAge

using Interpolations



T = 325.0   # total time
h = 0.01     # time step
Amax = 50.0 # Max infection age    
β = 0.06236946408226973
γ = 1/14
i0 = 6000.0
grid_t, grid_a = create_grid(T, h, Amax) # Creating Grid

g = zeros(length(grid_a))
g[1] = 1.0 / (grid_a[2] - grid_a[1])


dumb_weights = [0.05, 0.4, 0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
norm_weights = dumb_weights / sum(dumb_weights)
log_weights = log10.(norm_weights .* 1e7)
cutoff_age = 30.0
aw = collect(range(first(grid_a), cutoff_age, length=length(norm_weights)))
itp = linear_interpolation(aw, log_weights; extrapolation_bc = Throw())
lw_grid = itp.(grid_a[grid_a.<=cutoff_age])
s = map(ai -> ai <= cutoff_age ? itp(ai) : -Inf, grid_a)

β_prior = (mean = log(0.06236946408226973), sd = 0.5)
γ_prior = (mean = log(1/14), sd = 0.5)
i0_prior = (mean = log(6000.0), sd = 0.5)
σ_ww_prior = (mean = log(0.1), sd = 0.5)
R₀_prior = (mean = log(1), sd = 0.5)
σ_Rₜ_prior = (mean = log(0.01), sd = 0.5)

β_non_centered = rand()
γ_non_centered = rand()
i0_non_centered = rand()
σ_ww_non_centered = rand()
Rₜ_params_non_centered = rand(46+2)

obstimes_wastewater = collect(1.0:1.0:T)

likelihood_helper(
    obstimes_wastewater,
    grid_t,
    grid_a,
    g, 
    s,
    β_prior,
    γ_prior,
    i0_prior,
    σ_ww_prior,
    R₀_prior,
    σ_Rₜ_prior,
    β_non_centered, 
    γ_non_centered,
    i0_non_centered,
    σ_ww_non_centered,
    Rₜ_params_non_centered

)