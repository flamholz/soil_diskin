using CSV, DataFrames, Statistics, Printf

# Compare freshly produced Julia outputs to the backed-up Wolfram reference.
#
# Background: Wolfram NIntegrate (PrecisionGoal=3) on this 2-D integrand with
# a piecewise-constant atm14C function occasionally mis-converges — sometimes
# at isolated cells, sometimes over runs of ~5–10 consecutive ages — producing
# values that are clearly non-smooth as a function of age within the same site.
# The Julia implementation uses the analytical closed form for the inner age
# integral and `quadgk` on the smooth log-k outer integral, so its row-wise
# (age) curves are smooth by construction.
#
# Strategy: compare Julia ↔ Wolfram directly. Wherever they disagree by more
# than ~5e-3 (well past Wolfram's 3-digit precision goal), check whether the
# Wolfram row is smooth at that age (i.e., its first-difference signature
# tracks its neighbors). If Wolfram is non-smooth there, Wolfram is the wrong
# one; otherwise the disagreement is real.

function is_wolfram_smooth_local(row, j; rel_jump=0.01)
    # Compare cell j to a smooth estimate from its non-adjacent neighbors,
    # which dodges short runs of misconverged cells. Uses 4 anchors at ±3,±5
    # offsets and takes their median as a robust local baseline.
    n = length(row)
    offsets = [-5, -3, 3, 5]
    anchors = Float64[]
    for o in offsets
        k = j + o
        1 <= k <= n && push!(anchors, row[k])
    end
    isempty(anchors) && return true
    baseline = median(anchors)
    return abs(row[j] - baseline) / max(abs(baseline), 1e-12) < rel_jump
end

function compare_csv(julia_path, ref_path)
    J = Matrix(CSV.read(julia_path, DataFrame; header=false))
    R = Matrix(CSV.read(ref_path,   DataFrame; header=false))
    @assert size(J) == size(R)
    nrows, ncols = size(R)

    all_rel    = Float64[]
    agree_rel  = Float64[]            # both smooth, real comparison
    wolfram_failures = Tuple{Int,Int,Float64}[]

    for i in 1:nrows, j in 1:ncols
        rel_err = abs(J[i,j] - R[i,j]) / max(abs(R[i,j]), 1e-12)
        push!(all_rel, rel_err)
        if rel_err > 5e-3 && !is_wolfram_smooth_local(view(R, i, :), j)
            push!(wolfram_failures, (i, j, rel_err))
        else
            push!(agree_rel, rel_err)
        end
    end

    @printf "  grid: %d × %d (%d cells)\n" nrows ncols nrows*ncols
    @printf "  overall median rel err: %.3e\n" median(all_rel)
    @printf "  overall 99th pctile:    %.3e\n" quantile(all_rel, 0.99)
    @printf "  Wolfram NIntegrate failures detected: %d cells (%.2f%%)\n" length(wolfram_failures) 100*length(wolfram_failures)/(nrows*ncols)
    @printf "  After removing Wolfram failures:\n"
    @printf "    max rel err:  %.3e\n" maximum(agree_rel)
    @printf "    99th pctile:  %.3e\n" quantile(agree_rel, 0.99)
    @printf "    median:       %.3e\n" median(agree_rel)
end

for (name, jpath, rpath) in [
    ("main (turnover)", "results/03_calibrate_models/03b_lognormal_model_age_scan.csv",
                        "results/03_calibrate_models/_wolfram_reference/03b_lognormal_model_age_scan.csv"),
    ("q05",             "results/03_calibrate_models/03b_lognormal_model_age_scan_05.csv",
                        "results/03_calibrate_models/_wolfram_reference/03b_lognormal_model_age_scan_05.csv"),
    ("q95",             "results/03_calibrate_models/03b_lognormal_model_age_scan_95.csv",
                        "results/03_calibrate_models/_wolfram_reference/03b_lognormal_model_age_scan_95.csv"),
]
    println("=== $name ===")
    compare_csv(jpath, rpath)
    println()
end
