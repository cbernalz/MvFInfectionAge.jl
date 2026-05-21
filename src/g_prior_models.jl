@model function g_exp_prior_model(
    grid_a;
    α_prior = (mean = log(0.3), sd = 0.5),
    i0_prior = (mean = log(6000.0), sd = 0.5)
)
    h_a = grid_a[2] - grid_a[1]

    α_non_centered ~ Normal()
    i0_non_centered ~ Normal()

    α = exp(α_non_centered * α_prior.sd + α_prior.mean)
    i0 = exp(i0_non_centered * i0_prior.sd + i0_prior.mean)

    g = α .* exp.(-α .* grid_a)
    g = g ./ (h_a * sum(g))
    g_scaled = i0 .* g

    return (
        grid_a = grid_a,
        g = g_scaled,
        params = (
            α = α,
            i0 = i0
        )
    )

end


@model function g_uniform_prior_model(
    grid_a;
    i0_prior = (mean = log(6000.0), sd = 0.5)
)
    h_a = grid_a[2] - grid_a[1]

    i0_non_centered ~ Normal()
    i0 = exp(i0_non_centered * i0_prior.sd + i0_prior.mean)

    g = ones(length(grid_a))
    g = g ./ (h_a * sum(g))
    g_scaled = i0 .* g

    return (
        grid_a = grid_a,
        g = g_scaled,
        params = (
            i0 = i0,
        )
    )
end


@model function g_weighted_uniform_prior_model(
    grid_a;
    n_knots = 25,
    concentration = 1.0,
    i0_prior = (mean = log(6000.0), sd = 0.5)
)

    h_a = grid_a[2] - grid_a[1]

    i0_non_centered ~ Normal()
    i0 = exp(i0_non_centered * i0_prior.sd + i0_prior.mean)

    knot_idx = round.(Int, range(1, length(grid_a), length = n_knots))
    knot_a = grid_a[knot_idx]

    knot_weights ~ Dirichlet(fill(concentration, n_knots))

    interp_obj = linear_interpolation(
        knot_a,
        knot_weights;
        extrapolation_bc = Interpolations.Flat()
    )
    g_density = interp_obj.(grid_a)
    g_density .= max.(g_density, 0.0)
    g_density = g_density ./ (h_a * sum(g_density))
    g_scaled = i0 .* g_density

    return (
        grid_a = grid_a,
        g = g_scaled,
        params = (
            i0 = i0,
            knot_weights = knot_weights,
            knot_a = knot_a,
            concentration = concentration
        )
    )
end