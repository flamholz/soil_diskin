using QuadGK
using DifferentialEquations
using StaticArrays
using Distributions
using DataFrames
using CSV
using BenchmarkTools

"""
    Computes the derivative of the state variable `u` at time `t` given parameters `p`.

    Parameters:
    - `u`: State variable (SArray).
    - `p`: Tuple containing parameters (k, mu, sigma, input, fixed).
    - `t`: Current time.

    Returns:
    - SArray representing the derivative of `u` at time `t`.
    """
function dc_kt_dt(u, p, t)
    

    k, mu, sigma, input, fixed = p
    if fixed == true
        return SA[input .* pdf(LogNormal(mu, sigma), k) .- k .* u[1]]
    else
        ind = min(floor(Int,t+1),length(input))
        return SA[input[ind] * pdf(LogNormal(mu, sigma), k) - k * u[1]]
    end
end

 """
    Solves the ODE for a given `k`, `tau`, `age`, and `input`.
    Parameters:
    - `k`: Rate constant.
    - `tau`: Transit time.
    - `age`: Mean age.
    - `input`: Input value.
    - `fixed`: Boolean indicating if the input is fixed.
    - `tmax`: Maximum time for the ODE solution.
    - `ts_size`: Size of the time series.

    Returns:
    - ODE solution.
    """
function solve_ode(k,tau,age,input,fixed=true, tmax=100, ts_size=1000, X0 = 0)
   
    
    # Define the parameters for the LogNormal distribution
    sigma = sqrt(log(age/tau));
    mu = - log(sqrt(tau^3/age));

    # Create a tuple of parameters
    ps = (k,mu,sigma,input,fixed);
    tspan = (0.,tmax);
    
    # Initial condition
    u0 = SA[X0];
    # Define the ODE problem
    prob = ODEProblem(dc_kt_dt, u0,tspan , ps);
    # Solve the ODE problem

    ts = 10 .^ range(-1,log10(tmax),ts_size);
    return solve(prob,saveat=ts);
end

"""
    Runs the Diskin model for a given `tau`, `age`, `input`, and `fixed` flag.
    Parameters:
    - `tau`: Transit time.
    - `age`: Mean age.
    - `input`: Input value.
    - `fixed`: Boolean indicating if the input is fixed.
    - `tmax`: Maximum time for the ODE solution.
    - `ts_size`: Size of the time series.
    - `X0`: Initial condition (default is 0).

    Returns:
    - Result of the Diskin model.
    """
function run_diskin(tau,age,input,fixed=true, tmax=100., ts_size=1000, X0 = 0)
    res, error = quadgk(x -> solve_ode(x,tau,age,input,fixed, tmax, ts_size, X0), 0, Inf);
    return res;
end