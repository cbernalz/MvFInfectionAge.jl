"""
    mvf_infection_age_model(...)
This is the bayesian semi-parametric model for the McKendric-von Foerster infection age model.  


# Arguments
- `data_wastewater`: An array of log pathogen genome concentration in wastewater data.
- `obstime_wastewater`: An array of times (in days) for observed wastewater data.
- `param_change_points`: An array of times (in days) where Rₜ is allowed to change.
- `grid_t`: Time grid as a vector.  Can be created using the `grid` helper function.
- `grid_a`: Age grid as a vector.  Can be created using the `grid` helper function.
- `g`: defines how initial infected population is distributed over infection age ∫g(a) da = i0, where i0 is the initial number of infected individuals.  Currently not inferred and will be set to a point mass at a = 0.
- `s`: Shedding fuction over infection age.  
- `γ_prior`: An array of length 2 specifying the mean and standard deviation of the prior on γ on the log scale.  In the form (mean = ..., sd = ...).
- `i0_prior`: An array of length 2 specifying the mean and standard deviation of the prior on i0 on the log scale.  In the form (mean = ..., sd = ...).
- `σ_ww_prior`: An array of length 2 specifying the mean and standard deviation of the prior on σ_ww on the log scale.  In the form (mean = ..., sd = ...).
- `R₀_prior`: An array of length 2 specifying the mean and standard deviation of the prior on R₀ on the log scale.  In the form (mean = ..., sd = ...).
- `σ_Rₜ_prior`: An array of length 2 specifying the mean and standard deviation of the prior on σ_Rₜ on the log scale.  In the form (mean = ..., sd = ...).



"""
@model function mvf_infection_age_model(;
    data_wastewater,
    obstime_wastewater,
    s::Vector{Float64},
    Rₜ_prior_model,
    τ_prior_model,
    i0_prior = (mean = log(6000.0), sd = 0.5),
    σ_ww_prior = (mean = log(0.1), sd = 0.5),
    α_prior = (mean = log(0.3), sd = 0.5)
    )

        # PRIORS-----------------------------
        i0_non_centered ~ Normal()
        σ_ww_non_centered ~ Normal()
        α_non_centered ~ Normal()
        Rₜ_module ~ to_submodel(Rₜ_prior_model)
        τ_module ~ to_submodel(τ_prior_model)

        # TRANSFORMATIONS-----------------------------
        trans = likelihood_helper(
            obstime_wastewater,
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
        # Reject if the helper function failed and skip sample
        if !trans.success
            println("Likelihood Helper Failed for current parameters. Skipping sample...")
            Turing.@addlogprob! -Inf
            return
        end

        # Likelihood calculations------------
        for i in 1:length(obstime_wastewater)
            data_wastewater[i] ~ Normal(trans.log_W_means[i], trans.σ_ww)
        end

        return (
            log_W_means = trans.log_W_means, I_means = trans.I_means, 
            τ = trans.τ, c_t = trans.c_t, 
            g = trans.g, α = trans.α,
            Rₜ = trans.Rₜ,
            Rₜ_params = Rₜ_module.params,
            τ_params = trans.τ_params,
            i0 = trans.i0, σ_ww = trans.σ_ww
        )

    end