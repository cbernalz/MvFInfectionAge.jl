"""
    τ_exp_prior_model(...)
This function defines a prior model for the infection-age dependent shedding profile τ, where τ is modeled as an exponential decay function of the form τ(a) = exp(-γ * a), with a being the infection age. The parameter γ controls the rate of decay and has a log-normal prior.

# Arguments
- `grid_t`: Time grid as a vector.  Can be created using the `grid` helper function.
- `grid_a`: Age grid as a vector.  Can be created using the `grid` helper function.
- `γ_prior`: An array of length 2 specifying the mean and standard deviation of the prior on γ on the log scale.  In the form (mean = ..., sd = ...).

# Returns
- A named tuple containing:
    - `grid_t`: The input time grid.
    - `grid_a`: The input age grid.
    - `τ`: The shedding profile τ evaluated at each age in the grid.
    - `params`: A named tuple containing the parameter γ.

"""

@model function τ_exp_prior_model(
    grid_t,
    grid_a;
    γ_prior = (mean = log(0.2), sd = 0.5)
)
    γ_non_centered ~ Normal()
    γ = exp(γ_non_centered * γ_prior.sd + γ_prior.mean)

    τ = exp.(-γ .* grid_a)

    return (
        grid_t = grid_t,
        grid_a = grid_a,
        τ = τ,
        params = (γ = γ,)
    )
end



@model function τ_phase_type_prior_model(
    grid_t,
    grid_a;
    n_infection_comp = 7,
    exposed_rate_prior = (mean = log(1/7), sd = 0.25),
    infection_rate_prior = (mean = log(1/7), sd = 0.25)
)

    ## PRIORS-----------------------------
    exposed_rate_non_centered ~ Normal()
    infection_rate_non_centered ~ Normal()

    ## TRANSFORMATIONS-----------------------------
    exposed_rate = exp(exposed_rate_non_centered * exposed_rate_prior.sd + exposed_rate_prior.mean)
    infection_rate = exp(infection_rate_non_centered * infection_rate_prior.sd + infection_rate_prior.mean)

    nu = n_infection_comp * infection_rate

    Q = zeros(Float64, n_infection_comp + 1, n_infection_comp + 1)
    Q[1,1] = -ForwardDiff.value(exposed_rate)
    Q[1,2] = ForwardDiff.value(exposed_rate)
    for i in 2:(n_infection_comp + 1)
        Q[i,i] = -ForwardDiff.value(nu)
        if i < (n_infection_comp + 1)
            Q[i,i+1] = ForwardDiff.value(nu)
        end
    end
    
    init_vec = zeros(Float64, n_infection_comp + 1)
    init_vec[1] = 1.0

    infectious_selector = zeros(Float64, n_infection_comp + 1)
    infectious_selector[2:end] .= 1.0

    inf_prob = Vector{Float64}(undef, length(grid_a))
    τ = Vector{Float64}(undef, length(grid_a))
    for i in eachindex(grid_a)
        ai = grid_a[i]
        exp_Q = exp(Q * ai)
        inf_prob[i] = infectious_selector' * exp_Q * infectious_selector
        τ[i] = init_vec' * exp_Q * infectious_selector
    end
    inf_prob = inf_prob / sum(inf_prob * (grid_a[2] - grid_a[1]))
    
    return (
        grid_t = grid_t,
        grid_a = grid_a,
        τ = τ,
        inf_prob = inf_prob,
        params = (exposed_rate = exposed_rate, infection_rate = infection_rate)
    )
end


@model function τ_gamma_prior_model(
    grid_t,
    grid_a;
    μ_prior = (mean = log(7.0), sd = 0.5),
    σ_prior = (mean = log(1.5), sd = 0.5)
)
    h = grid_a[2] - grid_a[1]

    μ_non_centered ~ Normal()
    σ_non_centered ~ Normal()

    μ = exp(μ_non_centered * μ_prior.sd + μ_prior.mean)
    σ = exp(σ_non_centered * σ_prior.sd + σ_prior.mean)

    τ = (grid_a .^ ((μ^2 / σ^2) - 1)) .* exp.(-grid_a ./ (σ^2 / μ))
    τ = τ / (h * sum(τ))

    return (
        grid_t = grid_t,
        grid_a = grid_a,
        τ = τ,
        params = (μ = μ, σ = σ)
    )
end