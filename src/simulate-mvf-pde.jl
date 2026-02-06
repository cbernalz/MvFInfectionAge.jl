"""
Simulates the McKendric-von Foerster PDE using the backward-difference / method of characteristics scheme.

Arguments:
    t - Time grid as a vector.  Can be created using the `grid` helper function.
    a - Age grid as a vector.  Can be created using the `grid` helper function.
    τ - τ(a) rate function at which individuals with age a infect susceptibles ∫τ(a) da < ∞.  If nothing passed, function uses τ(a) = β * exp(-γ * a) for β = 0.6 and γ = 0.2.
    c - c(t) contact rate function at time t.  If nothing passed, function uses c(t) = 1 for all t.
    g - g(a) defines how initial infected population is distributed over infection age ∫g(a) da = i0, where i0 is the initial number of infected individuals.  If nothing passed, function uses a point mass at a = 0.

Returns: 
    U - Matrix of size (L, Nt) where L is the number of age steps and Nt is the number of time steps.  Rows correspond to age bins and columns correspond to time steps.
"""
function simulate_mvf_pde(
    t::Vector{Float64},
    a::Vector{Float64},
    τ = nothing,
    c = nothing,
    g = nothing,

)
    ## Throw error if τ, c, or g are not all nothing or not all given
    n_nothing = (τ === nothing) + (c === nothing) + (g === nothing)
    if n_nothing != 0 && n_nothing != 3
        throw(ArgumentError(
            "Either provide all of τ, c, and g, or provide none of them to use default values."
        ))
    end

    ## Setting default τ, c, g if none provided
    if τ === nothing && c === nothing && g === nothing
        τ, c, g = default_τcg(a, t)
    end

    h = a[2] - a[1] # step size
    Nt = length(t) # number of time steps
    L = length(a) # number of age steps

    ## Storage
    U = zeros(L, Nt) # Rows -> age bins, Columns -> time steps
    U[:,1] .= g # set initial condition

    ## Main
    u = copy(g) # current age distribution of infected
    for k in 2:Nt
        # boundary condition for age
        new_infections = c[k] * h * sum(@. τ * u)

        # backward scheme
        for j in L:-1:2
            u[j] = u[j-1]
        end

        # filling in boundary condition
        u[1] = new_infections

        # storing results
        U[:,k] .= u
    end

    return U
end