using Pkg
Pkg.instantiate()

using XLSX
using CSV
using DataFrames
using Statistics

include("diskin_utils.jl")

raw_site_data = DataFrame(XLSX.readtable("data/balesdent_2018/balesdent_2018_raw.xlsx", "Profiles", first_row=8));
raw_site_data[!, "Cref_0-100estim"] = replace(raw_site_data[!, "Cref_0-100estim"], "NA" => missing);
raw_site_data[!, "Ctotal_0-100estim"] = replace(raw_site_data[!, "Ctotal_0-100estim"], "NA" => missing);

ratios = raw_site_data[:,"Cref_0-100estim"] ./ raw_site_data[:, "Ctotal_0-100estim"];

site_params = CSV.read("results/03_calibrate_models/03b_lognormal_predictions_calcurve.csv", DataFrame);
mean_age = mean(site_params[:,"pred"]);
mean_turnover = mean(site_params[:,"turnover"]);

# calculate the percentiles of the ratios
J_ratio = quantile(skipmissing(ratios), [0.025, 0.25, 0.5, 0.75, 0.975]);

tmax = 100_000; # maximum time for the ODE solution
ts_size = 1000; # size of the time series


function f_new(J1, J2, tau1, tau2, age1, age2, tmax, ts_size)
    labeled = run_diskin(tau2, age2, 1, true, tmax, ts_size);
    unlabeled = tau1 .- run_diskin(tau1, age1, 1, false, tmax);
    f_new = J2 * labeled ./ (J2 * labeled .+ J1 .* unlabeled);
    return f_new
end

function f_new_mu_sigma(J1, J2, mu1, mu2, sigma1, sigma2, tmax, ts_size)
    tau1 = exp(-mu1 + 0.5 * sigma1^2);
    age1 = tau1 * exp(sigma1^2);
    tau2 = exp(-mu2 + 0.5 * sigma2^2);
    age2 = tau2 * exp(sigma2^2);
    labeled = run_diskin(tau2, age2, 1, true, tmax, ts_size);
    unlabeled = tau1 .- run_diskin(tau1, age1, 1, false, tmax);
    f_new = J2 * labeled ./ (J2 * labeled .+ J1 .* unlabeled);
    return f_new
end
# using ProgressBars
# for i in ProgressBar(1:size(params, 1))
#         # run the diskin model for each row in params
#         # results[i,:] = run_diskin(params[i,:turnover], params[i,:pred], 1, true, tmax, ts_size);
#     result = run_diskin(params[i,:turnover], params[i,:pred],1, true, tmax, ts_size);
#     results[i,:] = reduce(vcat, result);
    
# end

input_data = []
mu_data = []
sigma_data = []



for ratio in J_ratio
    old_mu = -log(sqrt(mean_turnover^3/mean_age));
    old_sigma = sqrt(log(mean_age/mean_turnover));
    new_mu = -log(ratio) + old_mu;
    new_sigma = sqrt(2 * log(ratio) + old_sigma^2); 
    
    fnew_input = f_new(1, ratio, mean_turnover, mean_turnover, mean_age, mean_age, tmax, ts_size);
    fnew_mu = f_new_mu_sigma(1, 1, old_mu, new_mu, old_sigma, old_sigma, tmax, ts_size);
    fnew_sigma = f_new_mu_sigma(1, 1, old_mu, old_mu, old_sigma, new_sigma, tmax, ts_size);

    push!(input_data, reduce(vcat, fnew_input));
    push!(mu_data, reduce(vcat, fnew_mu));
    push!(sigma_data, reduce(vcat, fnew_sigma));
end
col_names =string.(round.(J_ratio, digits=2));
ts = 10 .^ range(-1,log10(tmax),ts_size);

input_data = hcat(input_data...);
mu_data = hcat(mu_data...);
sigma_data = hcat(sigma_data...);

input_df = DataFrame(input_data, col_names);
mu_df = DataFrame(mu_data, col_names);
sigma_df = DataFrame(sigma_data, col_names);

input_df[!, :time] = ts;
mu_df[!, :time] = ts;
sigma_df[!, :time] = ts;

# Save the dataframes to CSV files
CSV.write("results/06_sensitivity_analysis/lognormal_input_data.csv", input_df);
CSV.write("results/06_sensitivity_analysis/lognormal_mu_data.csv", mu_df);
CSV.write("results/06_sensitivity_analysis/lognormal_sigma_data.csv", sigma_df);




# @btime cont = run_diskin(17,1000,0.25,true,false);
# @btime noncont = run_diskin(17,1000,0.25,true,true);
# cont'./noncont