"""
Creates a grid for time and infection age based on parameters T, Amax, and h.

Arguments:
    T - Total time to simulate.  Default is T = 100.0.
    h - Step size.  Default is h = 0.01.
    Amax - Maximum infection age retained.  Default is Amax = 50.0.

Returns:
    t - Time grid as a vector.
    a - Age grid as a vector.
"""
function create_grid(
    T::Float64 = 100.0,
    h::Float64 = 0.01,
    Amax::Float64 = 50.0,
)
    Nt = Int(round(T / h)) + 1 # number of time steps
    L = Int(round(Amax / h)) + 1 # number of age steps

    t = collect(0:h:T) # time grid
    a = collect(0:h:Amax) # age grid

    return t, a
    
end

"""
Creates default τ(a), c(t), and g(a) functions if none are provided.

Arguments:
    t - Time grid as a vector.
    a - Age grid as a vector.
    β - Infection rate parameter.  Default is β = 0.6.
    γ - Recovery rate parameter.  Default is γ = 0.2.
    i0 - Initial number of infected individuals.  Default is i0 = 1.0.

Returns:
    τ - τ(a) rate function at which individuals with age a infect susceptibles ∫τ(a) da < ∞.
    c - c(t) contact rate function at time t.
    g - g(a) defines how initial infected population is distributed over infection age ∫g(a) da = i0.
"""
function default_τcg(
    t::AbstractVector,
    a::AbstractVector,
    β::Float64 = 0.6,
    γ::Float64 = 0.2,
    i0::Float64 = 1.0,
)
    τ = β .* exp.(-γ .* a)
    c = ones(length(t))
    g = zeros(length(a))
    g[1] = 1.0 / (a[2] - a[1])
    g = i0 .* g

    return τ, c, g
    
end



"""
    ChainsCustomIndex(c::Chains, indices_to_keep::BitMatrix)

Reduce Chains object to only wanted indices. 

Function created by Damon Bayer. 
"""
function ChainsCustomIndex(c::Chains, indices_to_keep::BitMatrix)
    min_length = minimum(mapslices(sum, indices_to_keep, dims = 1))
  v = c.value
  new_v = copy(v.data)
  new_v_filtered = cat([new_v[indices_to_keep[:, i], :, i][1:min_length, :] for i in 1:size(v, 3)]..., dims = 3)
  aa = AxisArray(new_v_filtered; iter = v.axes[1].val[1:min_length], var = v.axes[2].val, chain = v.axes[3].val)

  Chains(aa, c.logevidence, c.name_map, c.info)
end