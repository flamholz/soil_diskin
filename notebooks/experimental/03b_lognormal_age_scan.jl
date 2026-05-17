using Pkg
Pkg.instantiate()

using CSV, DataFrames, Printf
using Base.Threads

include("lognormal_radiocarbon.jl")

# Julia port of `03b_lognormal_age_scan.wls`. Produces the same three CSVs:
#   results/03_calibrate_models/03b_lognormal_model_age_scan.csv         (turnover)
#   results/03_calibrate_models/03b_lognormal_model_age_scan_05.csv      (turnover_q05)
#   results/03_calibrate_models/03b_lognormal_model_age_scan_95.csv      (turnover_q95)
#
# Each output row = one site; each column = one age in `agelist`. No header.

const SITES_PATH = "results/all_sites_14C_turnover.csv"
const OUT_DIR    = "results/03_calibrate_models"

# agelist = 10^Range[3, 5.5, (5.5 - 3)/100]  (101 log-spaced ages)
const AGELIST = 10 .^ collect(range(3.0, 5.5, length=101))

"""
    scan_all(atm, taus, agelist; rtol)

For each (tau, age), predict the lognormal ¹⁴C activity ratio. Returns a
Matrix of size length(taus) × length(agelist). Parallelized over sites.
"""
function scan_all(atm::AtmC14, taus::AbstractVector{<:Real},
                  agelist::AbstractVector{<:Real}; rtol::Real=1e-4)
    nT, nA = length(taus), length(agelist)
    out = Matrix{Float64}(undef, nT, nA)
    progress = Threads.Atomic{Int}(0)
    @threads for i in 1:nT
        @inbounds for j in 1:nA
            out[i, j] = lognormal_radiocarbon(atm, taus[i], agelist[j]; rtol=rtol)
        end
        done = Threads.atomic_add!(progress, 1) + 1
        @printf "  [%3d / %3d] tau=%.4f done\n" done nT taus[i]
    end
    return out
end

function write_matrix_csv(path::AbstractString, M::AbstractMatrix)
    # Wolfram exports the raw matrix with no header — match that.
    open(path, "w") do io
        for i in 1:size(M, 1)
            join(io, view(M, i, :), ",")
            print(io, "\n")
        end
    end
end

# === Load inputs ===
atm = load_atm14c("data/14C_atm_annot.csv")
@printf "atm14C loaded: %d knots (a ≥ 0), mean_R tail = %.6f\n" length(atm.ages) atm.mean_R

sites = CSV.read(SITES_PATH, DataFrame)
@printf "Sites loaded: %d rows\n" nrow(sites)
@printf "Running with %d threads.\n\n" nthreads()

mkpath(OUT_DIR)

# === Scan 1: turnover (all sites) ===
println("=== Scanning turnover (all sites) ===")
taus = Float64.(sites.turnover)
@time results = scan_all(atm, taus, AGELIST)
out_main = joinpath(OUT_DIR, "03b_lognormal_model_age_scan.csv")
write_matrix_csv(out_main, results)
@printf "Wrote %s (%d × %d)\n\n" out_main size(results, 1) size(results, 2)

# === Scan 2 & 3: turnover_q05 and turnover_q95 (sites with non-missing q05) ===
backfilled = filter(row -> !ismissing(row.turnover_q05), sites)
@printf "Sites with non-missing turnover_q05: %d\n\n" nrow(backfilled)

println("=== Scanning turnover_q05 ===")
taus_05 = Float64.(backfilled.turnover_q05)
@time results_05 = scan_all(atm, taus_05, AGELIST)
out_05 = joinpath(OUT_DIR, "03b_lognormal_model_age_scan_05.csv")
write_matrix_csv(out_05, results_05)
@printf "Wrote %s (%d × %d)\n\n" out_05 size(results_05, 1) size(results_05, 2)

println("=== Scanning turnover_q95 ===")
taus_95 = Float64.(backfilled.turnover_q95)
@time results_95 = scan_all(atm, taus_95, AGELIST)
out_95 = joinpath(OUT_DIR, "03b_lognormal_model_age_scan_95.csv")
write_matrix_csv(out_95, results_95)
@printf "Wrote %s (%d × %d)\n" out_95 size(results_95, 1) size(results_95, 2)
