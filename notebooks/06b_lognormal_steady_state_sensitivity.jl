include("diskin_utils.jl")

using XLSX
using CSV
using DataFrames
using Statistics

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
# using ProgressBars
# for i in ProgressBar(1:size(params, 1))
#         # run the diskin model for each row in params
#         # results[i,:] = run_diskin(params[i,:turnover], params[i,:pred], 1, true, tmax, ts_size);
#     result = run_diskin(params[i,:turnover], params[i,:pred],1, true, tmax, ts_size);
#     results[i,:] = reduce(vcat, result);
    
# end

input_data = []
tau_data = []
age_data = []
for ratio in J_ratio
    fnew_input = f_new(1, ratio, mean_turnover, mean_turnover, mean_age, mean_age, tmax, ts_size);
    fnew_tau = f_new(1, 1, mean_turnover, mean_turnover * ratio, mean_age, mean_age, tmax, ts_size);
    fnew_age = f_new(1, 1, mean_turnover, mean_turnover, mean_age, mean_age * ratio, tmax, ts_size);

    push!(input_data, reduce(vcat, fnew_input));
    push!(tau_data, reduce(vcat, fnew_tau));
    push!(age_data, reduce(vcat, fnew_age));
end
col_names =string.(round.(J_ratio, digits=2));
ts = 10 .^ range(-1,log10(tmax),ts_size);

input_data = hcat(input_data...);
tau_data = hcat(tau_data...);
age_data = hcat(age_data...);

input_df = DataFrame(input_data, col_names);
tau_df = DataFrame(tau_data, col_names);
age_df = DataFrame(age_data, col_names);

input_df[!, :time] = ts;
tau_df[!, :time] = ts;
age_df[!, :time] = ts;

# Save the dataframes to CSV files
CSV.write("results/06_sensitivity_analysis/lognormal_input_data.csv", input_df);
CSV.write("results/06_sensitivity_analysis/lognormal_tau_data.csv", tau_df);
CSV.write("results/06_sensitivity_analysis/lognormal_age_data.csv", age_df);




# @btime cont = run_diskin(17,1000,0.25,true,false);
# @btime noncont = run_diskin(17,1000,0.25,true,true);
# cont'./noncont