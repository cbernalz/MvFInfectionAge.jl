"""

    fit(...)
Sampler function to fit the model.  

# Arguments
- `data_wastewater`: An array of log pathogen genome concentration in wastewater data.
- `dates_wastewater`: An array of dates for observed wastewater data.  If not passed, obstime_wastewater is required.
- `obstime_wastewater`: An array of times (in days) for observed wastewater data.  If not passed, dates_wastewater is required.
- `grid_t`: Time grid as a vector.  Can be created using the `grid` helper function.
- `grid_a`: Age grid as a vector.  Can be created using the `grid` helper function.
- `g`: defines how initial infected population is distributed over infection age ∫g(a) da = i0, where i0 is the initial number of infected individuals.  Currently not inferred and will be set to a point mass at a = 0.
- `s`: Shedding fuction over infection age.  
- `γ_prior`: An array of length 2 specifying the mean and standard deviation of the prior on γ on the log scale.  In the form (mean = ..., sd = ...).
- `i0_prior`: An array of length 2 specifying the mean and standard deviation of the prior on i0 on the log scale.  In the form (mean = ..., sd = ...).
- `σ_ww_prior`: An array of length 2 specifying the mean and standard deviation of the prior on σ_ww on the log scale.  In the form (mean = ..., sd = ...).
- `R₀_prior`: An array of length 2 specifying the mean and standard deviation of the prior on R₀ on the log scale.  In the form (mean = ..., sd = ...).
- `σ_Rₜ_prior`: An array of length 2 specifying the mean and standard deviation of the prior on σ_Rₜ on the log scale.  In the form (mean = ..., sd = ...).
- `priors_only`: Boolean indicating whether to sample from the prior only.  Default is false.
- `n_samples`: Number of samples to draw from the posterior.  Default is 500
- `n_chains`: Number of MCMC chains to run.  Default is 1.
- `n_discard_initial`: Number of initial samples to discard as burn-in.  Default is 0.
- `seed`: Random seed for reproducibility.  Default is 2024.
- `init_params`: Initial parameters for MCMC sampling.  Default is nothing, which will use the default initialization of the sampler.

"""


function fit(
    data_wastewater,
    dates_wastewater,
    obstime_wastewater,
    s,
    Rₜ_prior_model,
    τ_prior_model,
    i0_prior,
    σ_ww_prior,
    α_prior,
    priors_only::Bool=false,
    n_samples::Int64=500, n_chains::Int64=1,
    n_discard_initial::Int64=0, seed::Int64=2024,
    init_params=nothing
)

    # Obstime processing-----------------------------
    if dates_wastewater === nothing && obstime_wastewater === nothing
        throw(ArgumentError("Must provide either dates_wastewater or obstime_wastewater!!!"))
    elseif dates_wastewater !== nothing && obstime_wastewater !== nothing
        throw(ArgumentError("Must provide either dates_wastewater or obstime_wastewater, not both!!!"))
    elseif dates_wastewater !== nothing
        obstime_wastewater = Dates.value.(dates_wastewater .- minimum(dates_wastewater)) .+ 1
    end
    obstime_wastewater = convert(Vector{Float64}, obstime_wastewater)

    # Model and Sampling-----------------------------
    my_model = mvf_infection_age_model(
        data_wastewater = data_wastewater,
        obstime_wastewater = obstime_wastewater,
        s = s,
        Rₜ_prior_model = Rₜ_prior_model,
        τ_prior_model = τ_prior_model,
        i0_prior = i0_prior,
        σ_ww_prior = σ_ww_prior,
        α_prior = α_prior
    )
    # Sampling
    if priors_only
        Random.seed!(seed)
        samples = sample(my_model, Prior(), MCMCThreads(), 400, n_chains)
    else
        Random.seed!(seed)
        # Optimize
        if init_params === nothing
            samples = sample(my_model, NUTS(), MCMCThreads(), n_samples, n_chains, discard_initial = n_discard_initial)
        else
            samples = sample(my_model, NUTS(), MCMCThreads(), n_samples, n_chains, discard_initial = n_discard_initial, init_params = init_params)
        end
    end 

    return(samples)
end











