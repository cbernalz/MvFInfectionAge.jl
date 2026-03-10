"""
    mvf_infection_age_model(...)
This is the bayesian semi-parametric model for the McKendric-von Foerster infection age model.  


# Arguments
- `data_wastewater`: An array of log pathogen genome concentration in wastewater data.
- `dates_wastewater`: An array of dates for observed wastewater data.  If not passed, obstime_wastewater is required.
- `obstime_wastewater`: An array of times (in days) for observed wastewater data.  If not passed, dates_wastewater is required.
- `grid_t`: Time grid as a vector.  Can be created using the `grid` helper function.
- `grid_a`: Age grid as a vector.  Can be created using the `grid` helper function.
- `g`: defines how initial infected population is distributed over infection age ∫g(a) da = i0, where i0 is the initial number of infected individuals.  Currently not inferred and will be set to a point mass at a = 0.
- `s`: Shedding fuction over infection age.  
- `β_prior`: An array of length 2 specifying the mean and standard deviation of the prior on β on the log scale.  In the form (mean = ..., sd = ...).
- `γ_prior`: An array of length 2 specifying the mean and standard deviation of the prior on γ on the log scale.  In the form (mean = ..., sd = ...).
- `i0_prior`: An array of length 2 specifying the mean and standard deviation of the prior on i0 on the log scale.  In the form (mean = ..., sd = ...).
- `σ_ww_prior`: An array of length 2 specifying the mean and standard deviation of the prior on σ_ww on the log scale.  In the form (mean = ..., sd = ...).
- `R₀_prior`: An array of length 2 specifying the mean and standard deviation of the prior on R₀ on the log scale.  In the form (mean = ..., sd = ...).
- `σ_Rₜ_prior`: An array of length 2 specifying the mean and standard deviation of the prior on σ_Rₜ on the log scale.  In the form (mean = ..., sd = ...).



"""
@model function mvf_infection_age_model(;
    data_wastewater::Vector{Float64},
    dates_wastewater::Union{Vector{Date}, Nothing} = nothing,
    obstime_wastewater::Union{Vector{Float64}, Nothing} = nothing,
    grid_t::Vector{Float64},
    grid_a::Vector{Float64},
    g::Vector{Float64},
    s::Vector{Float64},
    β_prior = (mean = log(0.6), sd = 0.5),
    γ_prior = (mean = log(0.2), sd = 0.5),
    i0_prior = (mean = log(6000.0), sd = 0.5),
    σ_ww_prior = (mean = log(0.1), sd = 0.5),
    R₀_prior = (mean = log(2.5), sd = 0.5),
    σ_Rₜ_prior = (mean = log(0.1), sd = 0.5)
    )
    # Obstime processing-----------------------------
    if dates_wastewater === nothing && obstime_wastewater === nothing
        throw(ArgumentError("Must provide either dates_wastewater or obstime_wastewater!!!"))
    elseif dates_wastewater !== nothing && obstime_wastewater !== nothing
        throw(ArgumentError("Must provide either dates_wastewater or obstime_wastewater, not both!!!"))
    elseif dates_wastewater !== nothing
        weeks = trunc(Int, (maximum(dates_wastewater) - minimum(dates_wastewater)).value / 7.0)
        obstime_wastewater = Dates.value.(dates_wastewater .- minimum(dates_wastewater)) .+ 1
    else 
        weeks = trunc(Int, maximum(obstime_wastewater) / 7.0)
    end

        # PRIORS-----------------------------
        β_non_centered ~ Normal()
        γ_non_centered ~ Normal()
        i0_non_centered ~ Normal()
        σ_ww_non_centered ~ Normal()
        Rₜ_params_non_centered ~ MvNormal(zeros(weeks + 2), I) # +2 for sigma and init

        # TRANSFORMATIONS-----------------------------
        trans = likelihood_helpers(
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
        # Reject if the helper function failed and skip sample
        if !trans.success
            Turing.@addlogprob! -Inf
            return
        end

        # Likelihood calculations------------
        for i in 1:length(obstimes_wastewater)
            data_wastewater[i] ~ Normal(trans.log_W_means[i], trans.σ_ww)
        end

        return (
            log_W_means = trans.log_W_means, I_means = trans.I_means, 
            τ = trans.τ, c_t = trans.c_t, g = trans.g,
            Rₜ = trans.Rₜ, σ_Rₜ = trans.σ_Rₜ, β = trans.β, γ = trans.γ, i0 = trans.i0, σ_ww = trans.σ_ww
        )

    end