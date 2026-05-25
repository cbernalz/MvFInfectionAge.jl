"""
Driver for calculating quantities required for likelihood calculation.

Arguments:

"""

function likelihood_helper(
    obstimes_wastewater,
    s,
    σ_ww_prior,
    σ_ww_non_centered,
    Rₜ_module,
    pde_internals_module
)
    try
        h = pde_internals_module.grid_t[2] - pde_internals_module.grid_t[1]
        ww_inx = Int[]
        for x in obstimes_wastewater
            idx = round(Int, (x - pde_internals_module.grid_t[1]) / h) + 1
            if idx < 1 || idx > length(pde_internals_module.grid_t) || !isapprox(pde_internals_module.grid_t[idx], x; atol=1e-8)
                throw(ArgumentError("Observation time $x is not aligned with grid_t."))
            end
            push!(ww_inx, idx)
        end
        if any(x -> x === nothing, ww_inx)
            throw(ArgumentError("All obstimes_wastewater must be present in grid_t."))
        end
        if length(pde_internals_module.g) != length(pde_internals_module.grid_a)
            throw(ArgumentError("pde_internals_module.g must have same length as pde_internals_module.grid_a."))
        end

        σ_ww = exp(σ_ww_non_centered * σ_ww_prior.sd + σ_ww_prior.mean)

        τ = pde_internals_module.τ

        # c(t) from weekly Rt parameters
        h_a = pde_internals_module.grid_a[2] - pde_internals_module.grid_a[1]
        τ_integral = h_a * sum(τ)
        
        c = Rₜ_module.Rₜ ./ τ_integral

        c_itp = linear_interpolation(Rₜ_module.timebreaks, c; extrapolation_bc = Interpolations.Flat()) ## can also be Line() to follow the slope of the last two points instead of flat extrapolation; Interpolations is used for package issues
        c_t = c_itp.(pde_internals_module.grid_t)

        # infection probability
        inf_prob = pde_internals_module.inf_prob

        sim = simulate_mvf_pde(
            pde_internals_module.grid_t,
            pde_internals_module.grid_a,
            τ,
            c_t,
            pde_internals_module.g,
            ww_inx,
            inf_prob,
            s,
        )
        
        return (
            success = true,
            log_W_means = sim.log_W_means,
            I_means = sim.I_means,
            inf_prob = inf_prob,
            τ = τ,
            c_t = c_t,
            g = pde_internals_module.g,
            Rₜ = Rₜ_module.Rₜ,
            Rₜ_params = Rₜ_module.params,
            τ_params = pde_internals_module.τ_params,
            g_params = pde_internals_module.g_params,
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