using Pkg
Pkg.activate(".")

using LinearAlgebra
using InvertedIndices
using PowerModels, PGLib
using JuMP, Ipopt, Gurobi
using Printf

include("./src/squeeze_functions.jl")

# ==========================================================================
# Round-1 experiment: is the reduced global QP
#
#   max_mu  mu' * B * B' * mu
#   s.t.    A' * mu == 0
#           c' * mu == -1        (required normalization: h = -c'mu > 0)
#           mu >= 0
#
# tractable for Gurobi to solve to global optimality?
#
# This comes from analytically eliminating delta from the original
# nonconvex Farkas-lemma QCQP (Model 2):
#   min_{mu,delta} delta'delta
#   s.t. A'mu=0, mu>=0, mu'(B*delta+c) = 0
# For fixed mu, the inner minimization over delta (minimum-norm solution
# of the scalar linear constraint (mu'B)*delta = -mu'c = h) gives
#   inf_delta delta'delta = h^2 / (mu'*B*B'*mu),
# which is invariant to positive rescaling of mu, so we normalize h=1
# (i.e. c'mu=-1) and are left with maximizing mu'*B*B'*mu over the cone
# slice. Let gamma* be that optimal value. Then
#
#   rho* := 1/gamma*   is the infimum of delta'delta (squared attack norm).
#   The Euclidean attack norm is sqrt(rho*) = sqrt(1/gamma*).
# ==========================================================================

function solve_reduced_qp(A, B, c, Mvec; time_limit=60.0, mip_gap=1e-4)
    num_mu = size(A, 1)

    # feasibility of the required slice c'mu = -1 must hold (h = -c'mu > 0
    # is required by the derivation); if this slice is empty, stop and
    # report infeasibility rather than silently flipping the sign.
    feas_model = Model(Gurobi.Optimizer)
    set_silent(feas_model)
    @variable(feas_model, mu_f[1:num_mu] >= 0)
    @constraint(feas_model, A' * mu_f .== 0.0)
    @constraint(feas_model, dot(c, mu_f) == -1.0)
    @objective(feas_model, Min, 0.0)
    optimize!(feas_model)

    if termination_status(feas_model) != MOI.OPTIMAL
        return (feasible_slice = false, status = string(termination_status(feas_model)),
                runtime = NaN, gamma_inc = NaN, gamma_bd = NaN, gap = NaN, nodes = NaN,
                mu = nothing)
    end

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit", time_limit)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", mip_gap)

    @variable(model, 0 <= mu[i=1:num_mu] <= Mvec[i])
    @constraint(model, A' * mu .== 0.0)
    @constraint(model, dot(c, mu) == -1.0)
    Bmu = B' * mu
    @objective(model, Max, dot(Bmu, Bmu))

    optimize!(model)

    status  = string(termination_status(model))
    runtime = MOI.get(model, MOI.SolveTimeSec())
    nodes   = try
        MOI.get(model, MOI.NodeCount())
    catch
        NaN
    end

    if has_values(model)
        gamma_inc = objective_value(model)
        mu_star   = value.(mu)
    else
        gamma_inc = NaN
        mu_star   = nothing
    end
    gamma_bd = try
        objective_bound(model)
    catch
        NaN
    end

    gap = isnan(gamma_inc) || gamma_inc == 0 ? NaN : abs(gamma_bd - gamma_inc) / abs(gamma_inc)

    return (feasible_slice = true, status = status, runtime = runtime, gamma_inc = gamma_inc,
            gamma_bd = gamma_bd, gap = gap, nodes = nodes, mu = mu_star)
end

# ==========================================================================
# per-component LP bounds:
#   M[i] = max mu[i]  s.t.  A'mu=0, c'mu=-1, mu>=0
# for each i. This gives the tightest possible (non-arbitrary) finite upper
# bound implied by the feasible slice itself -- no Big-M guessing.
# An unbounded LP for a given i means that component of mu has no finite
# upper bound on this slice (reported separately, not silently truncated).
# ==========================================================================
function component_upper_bounds(A, c)
    num_mu = size(A, 1)
    Mvec = fill(NaN, num_mu)
    unbounded_idx = Int[]
    other_idx = Tuple{Int,String}[]

    for i in 1:num_mu
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        set_optimizer_attribute(model, "DualReductions", 0)  # force clear INFEASIBLE vs UNBOUNDED
        @variable(model, mu[1:num_mu] >= 0)
        @constraint(model, A' * mu .== 0.0)
        @constraint(model, dot(c, mu) == -1.0)
        @objective(model, Max, mu[i])
        optimize!(model)

        status = termination_status(model)
        if status == MOI.OPTIMAL
            Mvec[i] = objective_value(model)
        elseif status == MOI.DUAL_INFEASIBLE
            push!(unbounded_idx, i)
        else
            push!(other_idx, (i, string(status)))
        end
    end

    return Mvec, unbounded_idx, other_idx
end

# ==========================================================================
# run only case5_pjm
# ==========================================================================
case = "pglib_opf_case14_ieee.m"
println("="^78)
println("Case: $case")

network_data       = pglib(case)
basic_network_data = PowerModels.make_basic_network(network_data)
zero_nonlinear_costs!(basic_network_data)

pm_result, model, model_fl, sys = solve_dcopf(basic_network_data)
A, B, c = copy(sys[:A]), copy(sys[:B]), copy(sys[:c])

Mvec, unbounded_idx, other_idx = component_upper_bounds(A, c)

println()
println("="^78)
println("PER-COMPONENT LP BOUNDS: $case")
println("="^78)
num_mu = size(A, 1)
@printf("%-8s %-14s\n", "i", "M[i]")
for i in 1:num_mu
    if isnan(Mvec[i])
        @printf("%-8d %-14s\n", i, "n/a")
    else
        @printf("%-8d %-14.8g\n", i, Mvec[i])
    end
end

println()
if isempty(unbounded_idx)
    println("No unbounded components.")
else
    println("UNBOUNDED components (no finite upper bound on this slice): ", unbounded_idx)
end
if !isempty(other_idx)
    println("Components with unexpected LP status: ", other_idx)
end

if !isempty(unbounded_idx)
    error("Cannot proceed: components $unbounded_idx have no finite upper bound.")
end

# ==========================================================================
# rerun the reduced QP with the exact LP-derived bounds 0 <= mu[i] <= M[i]
# ==========================================================================
res = solve_reduced_qp(A, B, c, Mvec; time_limit=60.0)

println()
println("="^78)
println("BOUNDED REDUCED QP RESULT: $case")
println("="^78)

if !res.feasible_slice
    println("Slice {A'mu=0, mu>=0, c'mu=-1} is INFEASIBLE (status=$(res.status)).")
else
    rho = (res.gamma_inc isa Number && !isnan(res.gamma_inc) && res.gamma_inc > 0) ? 1.0 / res.gamma_inc : NaN
    attack_norm = isnan(rho) ? NaN : sqrt(rho)

    @printf("termination status = %s\n", res.status)
    @printf("gamma_inc          = %.8g\n", res.gamma_inc)
    @printf("gamma_bd           = %.8g\n", res.gamma_bd)
    @printf("relative gap       = %.6g\n", res.gap)
    @printf("node count         = %.0f\n", res.nodes)
    @printf("runtime            = %.4f s\n", res.runtime)
    println()
    @printf("rho = 1/gamma      = %.8g\n", rho)
    @printf("sqrt(rho)          = %.8g\n", attack_norm)

    println()
    println("--- feasibility verification (mu*) ---")
    mu_star = res.mu
    @printf("max(abs.(A' * mu*))  = %.6g\n", maximum(abs.(A' * mu_star)))
    @printf("c' * mu*             = %.8g\n", dot(c, mu_star))
    @printf("minimum(mu*)         = %.6g\n", minimum(mu_star))
end
