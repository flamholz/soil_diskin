using CSV
using DataFrames
using Distributions
using QuadGK
using Statistics

# Julia port of `03b_lognormal_age_scan.wls`.
#
# For a lognormal distribution of decay rates k with parameters (μ, σ), and a
# constant input, the steady-state radiocarbon signal is
#
#   r(τ, ā) = (1/E[1/k]) ∫_0^∞ ∫_0^∞
#              R(a) e^{-a/8267} pdf(LogNormal(μ,σ), k) e^{-k a} dk da,
#
# where R(a) is the atmospheric ¹⁴C activity ratio `a` years before year 2000
# and 8267 yr is the ¹⁴C mean life. The Wolfram code does this as a 2-D
# adaptive integral. Because R(a) is piecewise constant, the inner `a`
# integral has an exact analytical form, leaving a single smooth 1-D
# quadrature over k.
#
# Inner integral, with α = 1/8267 + k:
#   I(k) = Σ_i c_i (e^{-α a_i} - e^{-α a_{i+1}}) / α  +  c_∞ e^{-α a_N} / α,
# where atm14C = c_i on [a_i, a_{i+1}) and c_∞ on [a_N, ∞).
#
# Outer integral in log-k space (u = log k), where the lognormal becomes a
# Gaussian and the integrand is smooth:
#   r(τ, ā) = exp(μ - σ²/2) ∫ φ_N(u; μ, σ) I(e^u) du.

const C14_HALF_LIFE_TAU = 8267.0  # 14C mean life (years)

"""
    AtmC14(ages, fm, mean_R)

Piecewise-constant atmospheric ¹⁴C lookup. `ages` must be sorted ascending and
contain only non-negative values (only forward-in-time ages matter for the
soil integral). `mean_R` is the extrapolation value beyond the last age, used
to match Mathematica's `ExtrapolationHandler -> {(MeanR &)}`.
"""
struct AtmC14
    ages::Vector{Float64}
    fm::Vector{Float64}
    mean_R::Float64
end

"""
    load_atm14c(path="data/14C_atm_annot.csv")

Replicates the Mathematica setup:
- Read CSV with header
- Take columns 4 (`years_before_2000`) and 5 (`R_14C`)
- `mean_R = mean` of `R_14C` over the last 50_000 rows of the original (descending-year) order
- Order-0 interpolation, with `mean_R` for any out-of-range value
"""
function load_atm14c(path::AbstractString="data/14C_atm_annot.csv")
    df = CSV.read(path, DataFrame)
    raw_age = Float64.(df[:, 4])  # years_before_2000
    raw_fm  = Float64.(df[:, 5])  # R_14C

    # `C14DATA[[-50000 ;;]]` in Mathematica is the LAST 50_000 rows in the
    # original (file) order. Same set of values regardless of subsequent sort.
    mean_R = mean(raw_fm[end-49_999:end])

    # Sort by age ascending and keep only ages >= 0 (the integral runs over a ≥ 0).
    perm = sortperm(raw_age)
    a_sorted = raw_age[perm]
    f_sorted = raw_fm[perm]
    mask = a_sorted .>= 0
    return AtmC14(a_sorted[mask], f_sorted[mask], mean_R)
end

"""
    inner_integral(atm, α) -> ∫_0^∞ atm14C(a) e^{-α a} da

Analytical because atm14C is piecewise constant. O(N) in the number of knots,
allocation-free.
"""
@inline function inner_integral(atm::AtmC14, α::Float64)
    ages = atm.ages
    fm   = atm.fm
    N    = length(ages)
    acc  = 0.0
    prev_exp = exp(-α * ages[1])  # ages[1] == 0 → 1.0, but stay general
    @inbounds for i in 1:N-1
        cur_exp = exp(-α * ages[i+1])
        acc += fm[i] * (prev_exp - cur_exp)
        prev_exp = cur_exp
    end
    acc += atm.mean_R * prev_exp
    return acc / α
end

"""
    lognormal_radiocarbon(atm, tau, age; rtol=1e-4)

Predicted bulk-pool ¹⁴C activity ratio for a lognormal Diskin model with
transit time `tau` and mean age `age`. Mirrors `LognormalRadiocarbon` in
`03b_lognormal_age_scan.wls`.
"""
function lognormal_radiocarbon(atm::AtmC14, tau::Real, age::Real; rtol::Real=1e-4)
    sig = sqrt(log(age/tau))
    mu  = -log(sqrt(tau^3/age))
    normal = Normal(mu, sig)
    # In log-k space the integrand is a smooth Gaussian-weighted function.
    # Truncate at ±10σ — well past double precision.
    u_lo = mu - 10*sig
    u_hi = mu + 10*sig
    integrand = u -> pdf(normal, u) * inner_integral(atm, 1/C14_HALF_LIFE_TAU + exp(u))
    result, _ = quadgk(integrand, u_lo, u_hi; rtol=rtol)
    # Multiply by 1/E[1/k] = exp(μ - σ²/2)
    return result * exp(mu - sig^2/2)
end

"""
    scan_ages(atm, tau, agelist; rtol=1e-4) -> Vector{Float64}

Compute the predicted ¹⁴C ratio at each age in `agelist` for fixed `tau`.
"""
function scan_ages(atm::AtmC14, tau::Real, agelist::AbstractVector; rtol::Real=1e-4)
    return [lognormal_radiocarbon(atm, tau, a; rtol=rtol) for a in agelist]
end
