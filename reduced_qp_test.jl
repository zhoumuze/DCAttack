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
# case selection: pass a key as the first CLI arg, e.g.
#   julia --project=. reduced_qp_test.jl 24_ieee 60
# (second arg is the time limit in seconds, default 60). Defaults to 5_pjm.
# Every run's bounds + results are persisted under results/ so nothing is
# lost when this script is re-invoked for the next case.
# ==========================================================================
const CASE_MAP = Dict(
    "5_pjm"    => "pglib_opf_case5_pjm.m",
    "14_ieee"  => "pglib_opf_case14_ieee.m",
    "24_ieee"  => "pglib_opf_case24_ieee_rts.m",
    "30_as"    => "pglib_opf_case30_as.m",
    "57_ieee"  => "pglib_opf_case57_ieee.m",
    "60_c"     => "pglib_opf_case60_c.m",
    "118_ieee" => "pglib_opf_case118_ieee.m",
)

casekey    = length(ARGS) >= 1 ? ARGS[1] : "5_pjm"
time_limit = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 60.0
case       = CASE_MAP[casekey]

results_dir = "results"
isdir(results_dir) || mkdir(results_dir)

println("="^78)
println("Case: $case  (key=$casekey, time_limit=$(time_limit)s)")

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
res = solve_reduced_qp(A, B, c, Mvec; time_limit=time_limit)

println()
println("="^78)
println("BOUNDED REDUCED QP RESULT: $case")
println("="^78)

logpath = joinpath(results_dir, "reduced_qp_$(casekey).txt")
csvpath = joinpath(results_dir, "reduced_qp_summary.csv")

open(logpath, "w") do io
    println(io, "case = $case (key=$casekey)")
    println(io, "num_mu = $num_mu")
    println(io)
    println(io, "--- per-component LP bounds M[i] = max mu[i] s.t. A'mu=0, c'mu=-1, mu>=0 ---")
    for i in 1:num_mu
        println(io, "  i=$i  M[i]=$(Mvec[i])")
    end
    println(io)
    if isempty(unbounded_idx)
        println(io, "No unbounded components.")
    else
        println(io, "UNBOUNDED components: $unbounded_idx")
    end
    println(io)

    if !res.feasible_slice
        msg = "Slice {A'mu=0, mu>=0, c'mu=-1} is INFEASIBLE (status=$(res.status))."
        println("$msg")
        println(io, msg)
    else
        rho = (res.gamma_inc isa Number && !isnan(res.gamma_inc) && res.gamma_inc > 0) ? 1.0 / res.gamma_inc : NaN
        attack_norm = isnan(rho) ? NaN : sqrt(rho)
        mu_star  = res.mu
        max_AtMu = maximum(abs.(A' * mu_star))
        cmu      = dot(c, mu_star)
        min_mu   = minimum(mu_star)

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
        @printf("max(abs.(A' * mu*))  = %.6g\n", max_AtMu)
        @printf("c' * mu*             = %.8g\n", cmu)
        @printf("minimum(mu*)         = %.6g\n", min_mu)

        @printf(io, "termination status = %s\n", res.status)
        @printf(io, "gamma_inc          = %.8g\n", res.gamma_inc)
        @printf(io, "gamma_bd           = %.8g\n", res.gamma_bd)
        @printf(io, "relative gap       = %.6g\n", res.gap)
        @printf(io, "node count         = %.0f\n", res.nodes)
        @printf(io, "runtime            = %.4f s\n", res.runtime)
        println(io)
        @printf(io, "rho = 1/gamma      = %.8g\n", rho)
        @printf(io, "sqrt(rho)          = %.8g\n", attack_norm)
        println(io)
        println(io, "--- feasibility verification (mu*) ---")
        @printf(io, "max(abs.(A' * mu*))  = %.6g\n", max_AtMu)
        @printf(io, "c' * mu*             = %.8g\n", cmu)
        @printf(io, "minimum(mu*)         = %.6g\n", min_mu)

        header_needed = !isfile(csvpath)
        open(csvpath, "a") do cio
            if header_needed
                println(cio, "case,num_mu,status,time_limit_s,runtime_s,gamma_inc,gamma_bd,rel_gap,nodes,rho,sqrt_rho,max_AtMu,cTmu,min_mu")
            end
            println(cio, "$casekey,$num_mu,$(res.status),$time_limit,$(res.runtime),$(res.gamma_inc),$(res.gamma_bd),$(res.gap),$(res.nodes),$rho,$attack_norm,$max_AtMu,$cmu,$min_mu")
        end
    end
end

println()
println("Saved: $logpath")
println("Appended: $csvpath")
