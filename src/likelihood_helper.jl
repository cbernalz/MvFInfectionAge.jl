"""
Driver for calculating quantities required for likelihood calculation.

Arguments:

"""

function likelihood_helper(
    obstimes_wastewater,
    param_change_points,
    grid_t,
    grid_a,
    g, 
    s,
    γ_prior,
    i0_prior,
    σ_ww_prior,
    R₀_prior,
    σ_Rₜ_prior,
    γ_non_centered,
    i0_non_centered,
    σ_ww_non_centered,
    Rₜ_params_non_centered

)
    try
        h = grid_t[2] - grid_t[1]
        ww_inx = Int[]
        for x in obstimes_wastewater
            idx = round(Int, (x - grid_t[1]) / h) + 1
            if idx < 1 || idx > length(grid_t) || !isapprox(grid_t[idx], x; atol=1e-8)
                throw(ArgumentError("Observation time $x is not aligned with grid_t."))
            end
            push!(ww_inx, idx)
        end
        if any(x -> x === nothing, ww_inx)
            throw(ArgumentError("All obstimes_wastewater must be present in grid_t."))
        end

        σ_Rₜ_non_centered = Rₜ_params_non_centered[1]
        Rₜ_init_non_centered = Rₜ_params_non_centered[2]
        Rₜ_steps_non_centered = Rₜ_params_non_centered[3:end]

        γ = exp(γ_non_centered * γ_prior.sd + γ_prior.mean)
        i0 = exp(i0_non_centered * i0_prior.sd + i0_prior.mean)
        σ_ww = exp(σ_ww_non_centered * σ_ww_prior.sd + σ_ww_prior.mean)

        R₀ = exp(R₀_prior.mean + R₀_prior.sd * Rₜ_init_non_centered)
        σ_Rₜ = exp(σ_Rₜ_non_centered * σ_Rₜ_prior.sd + σ_Rₜ_prior.mean)

        τ = exp.(-γ .* grid_a)

        # c(t) from weekly Rt parameters
        h_a = grid_a[2] - grid_a[1]
        τ_integral = h_a * sum(τ)

        c_no_init = exp.(log(R₀) .+ cumsum(Rₜ_steps_non_centered) * σ_Rₜ) ./ τ_integral
        c₀ = R₀ / τ_integral
        c = vcat(c₀, c_no_init)

        c_itp = linear_interpolation(param_change_points, c; extrapolation_bc = Line())
        c_t = c_itp.(grid_t)

        Rₜ = c .* τ_integral
 
        # initial condition scaled by i0
        g_scaled = i0 .* g

        # precompute decorated-integral weights
        inf_weights = exp.(-γ .* grid_a)

        sim = simulate_mvf_pde(
            grid_t,
            grid_a,
            τ,
            c_t,
            g_scaled,
            ww_inx,
            inf_weights,
            s,
        )
        
        return (
            success = true,
            log_W_means = sim.log_W_means,
            I_means = sim.I_means,
            τ = τ,
            c_t = c_t,
            g = g_scaled,
            Rₜ = Rₜ,
            σ_Rₜ = σ_Rₜ,
            γ = γ,
            i0 = i0,
            σ_ww = σ_ww
        )

    catch e
        println("likelihood_helper failed:")
        showerror(stdout, e)
        println()
        Base.show_backtrace(stdout, catch_backtrace())
        println()
        return (success = false,)
    end

end