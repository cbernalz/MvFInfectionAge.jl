"""
Simulates the McKendric-von Foerster PDE using the backward-difference / method of characteristics scheme.

Arguments:
    t - Time grid as a vector.  Can be created using the `grid` helper function.
    a - Age grid as a vector.  Can be created using the `grid` helper function.
    τ - τ(a) rate function at which individuals with age a infect susceptibles ∫τ(a) da < ∞.  If nothing passed, function uses τ(a) = β * exp(-γ * a) for β = 0.6 and γ = 0.2.
    c - c(t) contact rate function at time t.  If nothing passed, function uses c(t) = 1 for all t.
    g - g(a) defines how initial infected population is distributed over infection age ∫g(a) da = i0, where i0 is the initial number of infected individuals.  If nothing passed, function uses a point mass at a = 0.
    ww_inx - Indices of wastewater observation times in the time grid.
    inf_prob - Probabilities for infection age distribution.
    s - Weights for shedding distribution.

Returns: 
    U - Matrix of size (L, Nt) where L is the number of age steps and Nt is the number of time steps.  Rows correspond to age bins and columns correspond to time steps.
"""
function simulate_mvf_pde(
    t::AbstractVector,
    a::AbstractVector,
    τ::AbstractVector,
    c::AbstractVector,
    g::AbstractVector,
    ww_inx::AbstractVector,
    inf_prob::AbstractVector,
    s::AbstractVector,

)
    h = a[2] - a[1]
    Nt = length(t)
    L = length(a)

    T = promote_type(
        eltype(t),
        eltype(a),
        eltype(τ),
        eltype(c),
        eltype(g),
        eltype(inf_prob),
        eltype(s)
    ) 


    # current age profile
    u = copy(g)

    # storage only for observation times
    n_obs = length(ww_inx)
    I_means = zeros(T, n_obs)
    log_W_means = zeros(T, n_obs)
    obs_lookup = Dict{Int,Int}()
    for (i, idx) in enumerate(ww_inx)
        obs_lookup[idx] = i
    end

    #if haskey(obs_lookup, 1)
    #    I_current = h * dot(u, inf_prob)
    #    W_current = h * dot(u, s)

    #    if W_current <= zero(T) || !isfinite(W_current)
    #        throw(ArgumentError("Non-positive or non-finite wastewater mean at time index 1."))
    #    end
    #    I_means[1] = I_current
    #    log_W_means[1] = log(W_current)
    #end


    for k in 2:Nt
        # boundary condition for age
        new_infections = c[k] * h * sum(@. τ * u)

        # backward scheme
        for j in L:-1:2
            u[j] = u[j-1]
        end

        # filling in boundary condition
        u[1] = new_infections

        if haskey(obs_lookup, k)
            obs_i = obs_lookup[k]

            I_current = h * dot(u, inf_prob)
            W_current = h * dot(u, s)

            if W_current == zero(T)
                @warn "Zero wastewater mean at time index $k; adding 1e-10."
            elseif !isfinite(W_current)
                throw(ArgumentError("Non-finite wastewater mean at time index $k."))
            end
            #if W_current <= zero(T) || !isfinite(W_current)
            #    throw(ArgumentError("Non-positive or non-finite wastewater mean at time index $k."))
            #end

            #if W_current <= zero(T) || !isfinite(W_current)
            #    println("k = ", k)
            #    println("minimum(u) = ", minimum(u))
            #    println("maximum(u) = ", maximum(u))
            #    println("any nonfinite u = ", any(!isfinite, u))
            #    println("minimum(s) = ", minimum(s))
            #    println("maximum(s) = ", maximum(s))
            #    println("any nonfinite s = ", any(!isfinite, s))
            #    println("minimum(c) = ", minimum(c))
            #    println("maximum(c) = ", maximum(c))
            #    println("new_infections = ", new_infections)
            #    println("W_current = ", W_current)
            #    plot(u, label="u")
            #    error("Debug stop")
            #end

            I_means[obs_i] = I_current
            log_W_means[obs_i] = log(W_current + 1e-10) # adding small constant to avoid log of zero
        end
    end

    return (I_means = I_means, log_W_means = log_W_means)

end