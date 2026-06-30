using Pkg
Pkg.instantiate()

using Printf

include("../diskin_utils.jl")
include("diskin_utils_fast.jl")

# Compare the closed-form/quadrature implementation against the original
# ODE-inside-quadgk implementation across a few (tau, age) regimes.
cases = [
    (10.0,  100.0),   # moderate spread
    (1.0,   50.0),    # narrow distribution (small age/tau)
    (50.0,  500.0),   # large absolute scales
    (5.0,   1000.0),  # very wide distribution
]

tmax    = 1000.0
ts_size = 100   # keep small — original is slow

println("Comparing run_diskin (ODE) vs run_diskin_fast (closed-form):")
println(repeat("-", 78))

all_ok = true
for (tau, age) in cases
    sigma = sqrt(log(age/tau))
    mu    = -log(sqrt(tau^3/age))

    t_orig = @elapsed res_orig_raw = run_diskin(tau, age, 1, true, tmax, ts_size)
    vals_orig = reduce(vcat, res_orig_raw)

    t_fast = @elapsed (_, vals_fast) = run_diskin_fast(tau, age, 1.0; tmax=tmax, ts_size=ts_size)

    abs_err = abs.(vals_orig .- vals_fast)
    rel_err = abs_err ./ max.(abs.(vals_orig), 1e-12)
    max_abs = maximum(abs_err)
    max_rel = maximum(rel_err)

    # Original `solve` uses DifferentialEquations.jl default reltol=1e-3, so we
    # cannot expect the original to match the closed-form to better than ~0.1%.
    # Treat 5e-3 as agreement.
    ok = max_rel < 5e-3
    global all_ok &= ok

    @printf "tau=%6.1f age=%6.1f  μ=%+6.3f σ=%5.3f  | orig=%6.2fs fast=%6.4fs speedup=%6.1fx  max|Δ|=%.2e max rel=%.2e %s\n" tau age mu sigma t_orig t_fast (t_orig/t_fast) max_abs max_rel (ok ? "OK" : "FAIL")
end

println(repeat("-", 78))

# Self-consistency: the closed-form should agree with itself across rtols,
# confirming the residual discrepancy above lives in the ODE solver, not here.
println("\nClosed-form self-consistency (rtol 1e-10 vs 1e-6):")
for (tau, age) in cases
    sigma = sqrt(log(age/tau))
    mu    = -log(sqrt(tau^3/age))
    ts = 10 .^ range(-1, log10(tmax), ts_size)

    dist = Normal(mu, sigma)
    u_lo = mu - sigma^2 - 10*sigma
    u_hi = mu + 10*sigma
    a = [quadgk(u -> exp(-u)*pdf(dist,u)*(-expm1(-exp(u)*t)), u_lo, u_hi; rtol=1e-10)[1] for t in ts]
    b = [quadgk(u -> exp(-u)*pdf(dist,u)*(-expm1(-exp(u)*t)), u_lo, u_hi; rtol=1e-6)[1] for t in ts]
    @printf "  tau=%6.1f age=%6.1f  max rel diff = %.2e\n" tau age maximum(abs.(a .- b) ./ max.(abs.(a), 1e-12))
end

println("\n" * (all_ok ? "All cases agree (within ODE solver tolerance)." : "Mismatch detected."))
exit(all_ok ? 0 : 1)
