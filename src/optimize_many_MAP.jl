"""
    optimize_many_MAP(model, n_reps = 100, top_n = 1, verbose = true)

Try n_reps different initializations to get MAP estimate.

Function by Damon Bayer

"""
function optimize_many_MAP(model, n_reps = 100, top_n = 1, verbose = true)
  lp_res = repeat([-Inf], n_reps)
  for i in eachindex(lp_res)
      if verbose
          println(i)
      end
      Random.seed!(i)
      try
          lp_res[i] = optimize(model, MAP(), LBFGS(linesearch = LineSearches.BackTracking())).lp
      catch
      end
  end
  eligible_indices = findall(.!isnan.(lp_res) .& isfinite.(lp_res))
  best_n_seeds =  eligible_indices[sortperm(lp_res[eligible_indices], rev = true)][1:top_n]

  map(best_n_seeds) do seed
    Random.seed!(seed)
    optimize(model, MAP(), LBFGS(linesearch = LineSearches.BackTracking())).values.array
  end
end


"""
    optimize_many_MAP2(model, n_reps = 100, top_n = 1, verbose = true)

Try n_reps different initializations to get MAP estimate.

Modified by Christian Bernal Zelaya
"""
function optimize_many_MAP2(model, n_reps=100, top_n=1, verbose=true)
    println("Optimizing initializations....")
    lp_res = repeat([-Inf], n_reps)
    for i in eachindex(lp_res)
        if verbose
            println("Trial: $i")
        end
        Random.seed!(i)
        try
            result = optimize(model, MAP(), LBFGS(linesearch=LineSearches.BackTracking()))
            lp_res[i] = result.lp
            if verbose
                println("Optimization successful for trial $i with log-probability: $(result.lp)")
            end
        catch e
            if verbose
                println("Optimization failed for trial $i")
                showerror(stdout, e)
                println()
                Base.show_backtrace(stdout, catch_backtrace())
                println()
            end
        end
    end
    eligible_indices = findall(.!isnan.(lp_res) .& isfinite.(lp_res))
    best_n_seeds = eligible_indices[sortperm(lp_res[eligible_indices], rev=true)][1:top_n]
    map(best_n_seeds) do seed
        Random.seed!(seed)
        result = optimize(
            model,
            MAP(),
            LBFGS(linesearch = LineSearches.BackTracking())
        )
        println("Optimization found for seed $seed with log-probability: $(result.lp)")
        return result.values.array
    end
end

"""
    optimize_many_MAP2_wrapper(...)

Wrapper function for optimize_many_MAP2 that uses model based on inputs to function.

Created by Christian Bernal Zelaya
"""
function optimize_many_MAP2_wrapper(
    data_wastewater,
    dates_wastewater,
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
    n_reps=100,
    top_n=1,
    verbose=true,
    warning_bool=true
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
    param_change_points = collect(1:7:(weeks*7)) # Weekly change points for Rₜ
    param_change_points = convert(Vector{Float64}, param_change_points)
    obstime_wastewater = convert(Vector{Float64}, obstime_wastewater)
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
    return optimize_many_MAP2(my_model, n_reps, top_n, verbose)
end

