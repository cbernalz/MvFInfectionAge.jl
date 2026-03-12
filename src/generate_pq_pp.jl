"""
    generate_pq_pp(...)
This generates quantities and a posterior predictive distribution for the bayesian semi-parametric model for the McKendric-von Foerster infection age model.

# Arguments
- `samples`: Samples from the posterior/prior distribution.
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
- `seed::Int64=2024`: Random seed for reproducibility.
- `forecast::Bool=false`: A boolean to indicate if forecasting is to be done.
- `forecast_weeks::Int64=4`: Number of weeks to forecast.

# Returns
- Posterior Quanteties and Posterior Predictive Distribution.
"""
function generate_pq_pp(
    samples,
    data_wastewater,
    obstime_wastewater,
    grid_t,
    grid_a,
    g,
    s,
    γ_prior,
    i0_prior,
    σ_ww_prior,
    R₀_prior,
    σ_Rₜ_prior;
    seed::Int64=2024,
    forecast::Bool=false, forecast_days::Int64=14
)

    # Obstime processing-----------------------------
    weeks = trunc(Int, maximum(obstime_wastewater) / 7.0)
    param_change_points = collect(1:7:(weeks*7)) # Weekly change points for Rₜ
    param_change_points = convert(Vector{Float64}, param_change_points)
    obstime_wastewater = convert(Vector{Float64}, obstime_wastewater)

    if forecast
        last_value = obstime_wastewater[end]
        obstime_wastewater = vcat(obstime_wastewater,(last_value+1):(last_value+forecast_days))
        missing_data_wastewater = repeat([missing], length(obstime_wastewater))
        data_wastewater = vcat(data_wastewater, repeat([data_wastewater[end]], forecast_days))
    else
        missing_data_wastewater = repeat([missing], length(data_wastewater))
    end
    my_model = mvf_infection_age_model(
        data_wastewater = data_wastewater,
        obstime_wastewater = obstime_wastewater,
        param_change_points = param_change_points,
        grid_t = grid_t,
        grid_a = grid_a,
        g = g,
        s = s,
        γ_prior = γ_prior,
        i0_prior = i0_prior,
        σ_ww_prior = σ_ww_prior,
        R₀_prior = R₀_prior,
        σ_Rₜ_prior = σ_Rₜ_prior
    )
    my_model_forecast_missing = mvf_infection_age_model(
        data_wastewater = missing_data_wastewater,
        obstime_wastewater = obstime_wastewater,
        param_change_points = param_change_points,
        grid_t = grid_t,
        grid_a = grid_a,
        g = g,
        s = s,
        γ_prior = γ_prior,
        i0_prior = i0_prior,
        σ_ww_prior = σ_ww_prior,
        R₀_prior = R₀_prior,
        σ_Rₜ_prior = σ_Rₜ_prior
    )

    samples_df = DataFrame(samples)

    indices_to_keep = .!isnothing.(generated_quantities(my_model, samples))
    samples_randn = ChainsCustomIndex(samples, indices_to_keep)

    Random.seed!(seed)
    predictive_randn = predict(my_model_forecast_missing, samples_randn)
    Random.seed!(seed)
    println("Generating quantities...")
    gq_randn = generated_quantities(my_model, samples_randn)
    results = [DataFrame(predictive_randn), gq_randn, samples_df]

    return(results)

end


