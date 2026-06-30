using Pkg
Pkg.instantiate()

using CSV, DataFrames, Statistics, Printf

include("lognormal_radiocarbon.jl")

# Reference output produced by `03b_lognormal_age_scan.wls`. Each row is one
# site; columns correspond to ages in `agelist`.
wolfram_path = "results/03_calibrate_models/03b_lognormal_model_age_scan.csv"
sites_path   = "results/all_sites_14C_turnover.csv"

# Match Wolfram's `agelist = 10^Range[3, 5.5, (5.5 - 3)/100]` (101 points).
agelist = 10 .^ collect(range(3.0, 5.5, length=101))

ref = CSV.read(wolfram_path, DataFrame; header=false)
ref_mat = Matrix(ref)  # 99 sites × 101 ages

sites = CSV.read(sites_path, DataFrame)
turnovers = Float64.(sites.turnover)  # 99 sites
@assert length(turnovers) == size(ref_mat, 1)
@assert length(agelist)   == size(ref_mat, 2)

# Set up atmospheric 14C lookup once.
atm = load_atm14c("data/14C_atm_annot.csv")
@printf "atm14C loaded: %d non-negative-age knots, mean_R (tail) = %.6f\n" length(atm.ages) atm.mean_R

# Wolfram NIntegrate (PrecisionGoal=3) occasionally produces a spurious spike
# at a single age when adaptive 2-D quadrature mis-resolves the piecewise-
# constant inner integrand. Detect those by checking each cell against the
# average of its two age-neighbors in the same row.
function is_wolfram_outlier(row, ai; tol=0.02)
    1 < ai < length(row) || return false
    neighbor = 0.5 * (row[ai-1] + row[ai+1])
    return abs(row[ai] - neighbor) / max(abs(neighbor), 1e-12) > tol
end

# Evaluate a subset; full grid (99×101) is too slow with rtol=1e-5.
site_subset = [1, 2, 3, 10, 25, 50, 75, 99]
age_subset  = [1, 11, 26, 51, 76, 101]   # 10^3, 10^3.25, 10^3.625, 10^4.25, 10^4.875, 10^5.5

println("\nComparing Julia port vs saved Wolfram results (PrecisionGoal=3 ≈ 1e-3 reference accuracy):")
println(repeat("-", 100))
@printf "%4s %12s %10s %14s %14s %12s %12s %s\n" "site" "tau" "age" "wolfram" "julia" "abs_err" "rel_err" "note"

rel_errs = Float64[]
n_outliers = 0
for si in site_subset
    tau = turnovers[si]
    row = view(ref_mat, si, :)
    for ai in age_subset
        age = agelist[ai]
        wref = ref_mat[si, ai]
        jval = lognormal_radiocarbon(atm, tau, age; rtol=1e-5)
        abs_err = abs(jval - wref)
        rel_err = abs_err / max(abs(wref), 1e-12)
        outlier = is_wolfram_outlier(row, ai)
        note = outlier ? "(Wolfram outlier)" : ""
        if outlier
            global n_outliers += 1
        else
            push!(rel_errs, rel_err)
        end
        @printf "%4d %12.4f %10.1f %14.8f %14.8f %12.2e %12.2e %s\n" si tau age wref jval abs_err rel_err note
    end
end

println(repeat("-", 100))
max_rel = maximum(rel_errs)
med_rel = median(rel_errs)
@printf "Compared %d clean pairs (+ %d cells flagged as Wolfram NIntegrate outliers, skipped).\n" length(rel_errs) n_outliers
@printf "  max relative error: %.2e\n" max_rel
@printf "  median relative err: %.2e\n" med_rel
println(max_rel < 5e-3 ?
    "OK — Julia port agrees with Wolfram reference within its 3-digit precision goal." :
    "MISMATCH — relative error exceeds 5e-3.")
exit(max_rel < 5e-3 ? 0 : 1)
