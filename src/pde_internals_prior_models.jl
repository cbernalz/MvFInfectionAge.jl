@model function τ_gamma_g_exp_prior_model(
    grid_t,
    grid_a;
    μ_prior = (mean = log(7.0), sd = 0.5),
    σ_prior = (mean = log(1.5), sd = 0.5),
    α_prior = (mean = log(0.3), sd = 0.5),
    i0_prior = (mean = log(6000.0), sd = 0.5)
)
    h_a = grid_a[2] - grid_a[1]

    ## τ PRIORS ---------------------------------------------------------
    μ_non_centered ~ Normal()
    σ_non_centered ~ Normal()

    μ = exp(μ_non_centered * μ_prior.sd + μ_prior.mean)
    σ = exp(σ_non_centered * σ_prior.sd + σ_prior.mean)

    ## τ CONSTRUCTION --------------------------------------------------
    grid_a_safe = max.(grid_a, eps())

    τ = (grid_a_safe .^ ((μ^2 / σ^2) - 1)) .* exp.(-grid_a_safe ./ (σ^2 / μ))
    τ = τ ./ (sum(τ) * h_a)

    cdf = cumsum(τ) .* h_a
    inf_prob = 1 .- cdf
    inf_prob = vcat(1.0, inf_prob[1:end-1])

    ## g PRIORS --------------------------------------------------------
    α_non_centered ~ Normal()
    i0_non_centered ~ Normal()

    α = exp(α_non_centered * α_prior.sd + α_prior.mean)
    i0 = exp(i0_non_centered * i0_prior.sd + i0_prior.mean)

    ## g CONSTRUCTION --------------------------------------------------
    g = α .* exp.(-α .* grid_a)
    g = g ./ (h_a * sum(g .* inf_prob))
    g_scaled = i0 .* g

    return (
        grid_t = grid_t,
        grid_a = grid_a,

        τ = τ,
        inf_prob = inf_prob,
        g = g_scaled,

        τ_params = (
            μ = μ,
            σ = σ
        ),
        g_params = (
            α = α,
            i0 = i0
        )
    )
end


@model function τ_gamma_g_gamma_prior_model(
    grid_t,
    grid_a;
    τ_μ_prior = (mean = log(7.0), sd = 0.5),
    τ_σ_prior = (mean = log(1.5), sd = 0.5),
    g_μ_prior = (mean = log(7.0), sd = 0.5),
    g_σ_prior = (mean = log(5.0), sd = 0.5),
    i0_prior = (mean = log(6000.0), sd = 0.5)
)
    h_a = grid_a[2] - grid_a[1]

    ## τ PRIORS ---------------------------------------------------------
    τ_μ_non_centered ~ Normal()
    τ_σ_non_centered ~ Normal()

    τ_μ = exp(τ_μ_non_centered * τ_μ_prior.sd + τ_μ_prior.mean)
    τ_σ = exp(τ_σ_non_centered * τ_σ_prior.sd + τ_σ_prior.mean)

    ## τ CONSTRUCTION --------------------------------------------------
    grid_a_safe = max.(grid_a, eps())

    τ = (grid_a_safe .^ ((τ_μ^2 / τ_σ^2) - 1)) .* exp.(-grid_a_safe ./ (τ_σ^2 / τ_μ))
    τ = τ ./ (sum(τ) * h_a)

    cdf = cumsum(τ) .* h_a
    inf_prob = 1 .- cdf
    inf_prob = vcat(1.0, inf_prob[1:end-1])

    ## g PRIORS --------------------------------------------------------
    g_μ_non_centered ~ Normal()
    g_σ_non_centered ~ Normal()
    i0_non_centered ~ Normal()

    g_μ = exp(g_μ_non_centered * g_μ_prior.sd + g_μ_prior.mean)
    g_σ = exp(g_σ_non_centered * g_σ_prior.sd + g_σ_prior.mean)
    i0 = exp(i0_non_centered * i0_prior.sd + i0_prior.mean)

    ## g CONSTRUCTION --------------------------------------------------
    g = (grid_a_safe .^ ((g_μ^2 / g_σ^2) - 1)) .* exp.(-grid_a_safe ./ (g_σ^2 / g_μ))
    g = g ./ (h_a * sum(g .* inf_prob))
    g_scaled = i0 .* g

    return (
        grid_t = grid_t,
        grid_a = grid_a,

        τ = τ,
        inf_prob = inf_prob,
        g = g_scaled,

        τ_params = (
            μ = τ_μ,
            σ = τ_σ
        ),
        g_params = (
            μ = g_μ,
            σ = g_σ,
            i0 = i0
        )
    )
end