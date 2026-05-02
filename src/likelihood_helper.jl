"""
Driver for calculating quantities required for likelihood calculation.

Arguments:

"""

function likelihood_helper(
    obstimes_wastewater,
    s,
    i0_prior,
    σ_ww_prior,
    α_prior,
    i0_non_centered,
    σ_ww_non_centered,
    α_non_centered,
    Rₜ_module,
    τ_module

)
    try
        h = τ_module.grid_t[2] - τ_module.grid_t[1]
        ww_inx = Int[]
        for x in obstimes_wastewater
            idx = round(Int, (x - τ_module.grid_t[1]) / h) + 1
            if idx < 1 || idx > length(τ_module.grid_t) || !isapprox(τ_module.grid_t[idx], x; atol=1e-8)
                throw(ArgumentError("Observation time $x is not aligned with grid_t."))
            end
            push!(ww_inx, idx)
        end
        if any(x -> x === nothing, ww_inx)
            throw(ArgumentError("All obstimes_wastewater must be present in grid_t."))
        end

        i0 = exp(i0_non_centered * i0_prior.sd + i0_prior.mean)
        σ_ww = exp(σ_ww_non_centered * σ_ww_prior.sd + σ_ww_prior.mean)
        α = exp(α_non_centered * α_prior.sd + α_prior.mean)

        τ = τ_module.τ

        # c(t) from weekly Rt parameters
        h_a = τ_module.grid_a[2] - τ_module.grid_a[1]
        τ_integral = h_a * sum(τ)
        
        c = Rₜ_module.Rₜ ./ τ_integral

        c_itp = linear_interpolation(Rₜ_module.timebreaks, c; extrapolation_bc = Interpolations.Flat()) ## can also be Line() to follow the slope of the last two points instead of flat extrapolation; Interpolations is used for package issues
        c_t = c_itp.(τ_module.grid_t)
 
        # initial condition scaled by i0
        #f(α) = h * sum(exp.(-α .* τ_module.grid_a) .* ForwardDiff.value.(τ)) - 1.0
        #α = bisect_root(f, 1e-8, 1.0)
        #α = fzero(f, 0.1)
        #α = 0.3
        g = α .* exp.(-α .* τ_module.grid_a)
        g_scaled = i0 .* g

        # infection probability
        inf_prob = τ_module.inf_prob

        sim = simulate_mvf_pde(
            τ_module.grid_t,
            τ_module.grid_a,
            τ,
            c_t,
            g_scaled,
            ww_inx,
            inf_prob,
            s,
        )
        
        return (
            success = true,
            log_W_means = sim.log_W_means,
            I_means = sim.I_means,
            τ = τ,
            c_t = c_t,
            g = g_scaled,
            α = α,
            Rₜ = Rₜ_module.Rₜ,
            Rₜ_params = Rₜ_module.params,
            τ_params = τ_module.params,
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