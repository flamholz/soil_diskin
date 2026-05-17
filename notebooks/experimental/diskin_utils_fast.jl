using QuadGK
using Distributions

"""
    run_diskin_fast(tau, age, input=1.0; tmax=100.0, ts_size=1000)

Closed-form evaluation of the lognormal Diskin model for a constant input.

For each rate k, the linear ODE  dC_k/dt = I*f(k) - k*C_k  with C_k(0)=0 has
solution C_k(t) = I*f(k)/k * (1 - exp(-k t)). Integrating over k gives

    C(t) = I ∫₀^∞ f(k)/k (1 - exp(-k t)) dk
         = I ∫ exp(-u) φ_N(u; μ, σ) (1 - exp(-exp(u) t)) du,   u = log k

which is a single smooth 1-D quadrature per time point — no ODE solver.

Returns (ts, C) with ts log-spaced from 0.1 to tmax (matching `run_diskin`).
"""
function run_diskin_fast(tau, age, input=1.0; tmax=100.0, ts_size=1000)
    sigma = sqrt(log(age/tau))
    mu    = -log(sqrt(tau^3/age))
    ts    = 10 .^ range(-1, log10(tmax), ts_size)
    return ts, diskin_C_of_t(ts, mu, sigma, input)
end

"""
    diskin_C_of_t(ts, mu, sigma, input=1.0)

Evaluate C(t) at each t in `ts` for the lognormal Diskin model with
parameters (μ, σ) of log-rate and constant input.
"""
function diskin_C_of_t(ts::AbstractVector, mu::Real, sigma::Real, input::Real=1.0)
    dist = Normal(mu, sigma)
    # The integrand exp(-u) φ_N(u; μ, σ) peaks at u = μ - σ², so widen the
    # lower bound accordingly. ±10σ is well beyond machine precision.
    u_lo = mu - sigma^2 - 10*sigma
    u_hi = mu + 10*sigma
    results = Vector{Float64}(undef, length(ts))
    for (i, t) in enumerate(ts)
        # Use -expm1(-x) instead of 1 - exp(-x) for accuracy when x is small.
        f = u -> exp(-u) * pdf(dist, u) * (-expm1(-exp(u) * t))
        val, _ = quadgk(f, u_lo, u_hi; rtol=1e-10)
        results[i] = input * val
    end
    return results
end
