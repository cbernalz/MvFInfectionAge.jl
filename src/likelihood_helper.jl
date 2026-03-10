"""
Driver for calculating quantities required for likelihood calculation.

Arguments:

"""

function likelihood_helper(
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
    try

        ww_inx = [findfirst(x .== grid_t) for x in collect(Float64, obstimes_wastewater)]
        if any(x -> x === nothing, ww_inx)
            throw(ArgumentError("All obstimes_wastewater must be present in grid_t."))
        end

        σ_Rₜ_non_centered = Rₜ_params_non_centered[1]
        Rₜ_init_non_centered = Rₜ_params_non_centered[2]
        log_Rₜ_steps_non_centered = Rₜ_params_non_centered[3:end]

        β = exp(β_non_centered * β_prior.sd + β_prior.mean)
        γ = exp(γ_non_centered * γ_prior.sd + γ_prior.mean)
        i0 = exp(i0_non_centered * i0_prior.sd + i0_prior.mean)
        σ_ww = exp(σ_ww_non_centered * σ_ww_prior.sd + σ_ww_prior.mean)

        R₀ = exp(R₀_prior.mean + R₀_prior.sd * Rₜ_init_non_centered)
        σ_Rₜ = exp(σ_Rₜ_non_centered * σ_Rₜ_prior.sd + σ_Rₜ_prior.mean)

        τ = β .* exp.(-γ .* grid_a)
        c_no_init = exp.(log(R₀) .+ cumsum(log_Rₜ_steps_non_centered) * σ_Rₜ) / ( (grid_a[2] - grid_a[1]) * sum(τ))
        c₀ = R₀ / ( (grid_a[2] - grid_a[1]) * sum(τ))
        c = vcat(c₀, c_no_init)
        c_itp = linear_interpolation(collect(1:length(c)), c; extrapolation_bc=Line())
        c_t = c_itp.(grid_t)
        Rₜ = c .* ((grid_a[2] - grid_a[1]) * sum(τ))

        g = i0 .* g

        U = simulate_mvf_pde(
            grid_t,
            grid_a,
            τ,
            c_t,
            g,
        )

        # Using Mckendrick-von Foerster decorated integral
        I = zeros(length(grid_t))
        for k in 1:length(grid_t)
            I[k] = (grid_a[2] - grid_a[1]) * sum(U[:, k] .* exp.(@. -γ * grid_a))
        end
        I_means = I[ww_inx]
        # Log wastewater decorated integral
        W = zeros(length(grid_t))
        for k in 1:length(grid_t)
            W[k] = (grid_a[2] - grid_a[1]) * sum(@. U[:, k] * (10 .^ s))
        end
        log_W = log.(W)
        log_W_means = log_W[ww_inx]

        return (success = true, 
        log_W_means = log_W_means, I_means = I_means, 
        τ = τ, c_t = c_t, g = g,
        Rₜ = Rₜ, σ_Rₜ = σ_Rₜ, β = β, γ = γ, i0 = i0, σ_ww = σ_ww)

    catch e
        return (success = false,)
    end

end