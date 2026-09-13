using Pkg
Pkg.activate(".")

using LinearAlgebra
using InvertedIndices
using PowerModels, PGLib
using JuMP, Ipopt, Gurobi
using Printf

include("./src/squeeze_functions.jl")

# ==========================================================================
# Controlled experiment: y-space reformulation of the reduced global QP.
#
# OLD (mu-space, see reduced_qp_test.jl -- NOT modified here):
#   max_mu  mu' * B * B' * mu
#   s.t.    A'mu = 0, c'mu = -1, mu >= 0, 0 <= mu <= M   (M from LP bounds)
#
# NEW (y-space, this file):
#   y := B'*mu  (n_delta-dimensional)
#   max_{mu,y}  sum_k y[k]^2
#   s.t.        y == B'*mu
#               A'mu = 0, c'mu = -1, mu >= 0, 0 <= mu <= M
#               L[k] <= y[k] <= U[k]   (L,U from LP bounds on (B'mu)[k])
#
# These are mathematically identical programs (y is just a linear
# substitution). The question is purely computational: does exposing y
# explicitly give Gurobi a stronger root relaxation / faster bound
# convergence than expanding mu'*B*B'*mu into |mu|^2 bilinear cross terms?
#
# This script does NOT touch reduced_qp_test.jl or results/reduced_qp_*.
# All outputs go to results/yspace_reduced_qp_*.txt/csv and
# results/yspace_gurobi_*.log.
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
const ALL_CASES = ["5_pjm", "14_ieee", "24_ieee", "30_as", "57_ieee", "60_c", "118_ieee"]

results_dir = "results"
isdir(results_dir) || mkdir(results_dir)

# --------------------------------------------------------------------------
# mu-space componentwise LP bounds: M[i] = max mu[i] s.t. A'mu=0, c'mu=-1, mu>=0
# (duplicated here, verbatim in spirit, from reduced_qp_test.jl so this file
# is self-contained and the original is left untouched)
# --------------------------------------------------------------------------
function component_upper_bounds(A, c)
    num_mu = size(A, 1)
    Mvec = fill(NaN, num_mu)
    unbounded_idx = Int[]
    other_idx = Tuple{Int,String}[]

    for i in 1:num_mu
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        set_optimizer_attribute(model, "DualReductions", 0)
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

# --------------------------------------------------------------------------
# y-space componentwise LP bounds, reusing ONE JuMP model across all
# 2*n_delta LP solves (only the objective is changed between solves).
#   L[k] = min_mu (B'mu)[k]  s.t. A'mu=0, c'mu=-1, mu>=0
#   U[k] = max_mu (B'mu)[k]  s.t. A'mu=0, c'mu=-1, mu>=0
# --------------------------------------------------------------------------
function compute_y_bounds(A, B, c)
    num_mu  = size(A, 1)
    n_delta = size(B, 2)
    L = fill(NaN, n_delta)
    U = fill(NaN, n_delta)
    unbounded_L = Int[]
    unbounded_U = Int[]
    other_L = Tuple{Int,String}[]
    other_U = Tuple{Int,String}[]

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "DualReductions", 0)
    @variable(model, mu[1:num_mu] >= 0)
    @constraint(model, A' * mu .== 0.0)
    @constraint(model, dot(c, mu) == -1.0)

    for k in 1:n_delta
        expr = dot(B[:, k], mu)  # (B'mu)[k]

        @objective(model, Max, expr)
        optimize!(model)
        st = termination_status(model)
        if st == MOI.OPTIMAL
            U[k] = objective_value(model)
        elseif st == MOI.DUAL_INFEASIBLE
            push!(unbounded_U, k)
        else
            push!(other_U, (k, string(st)))
        end

        @objective(model, Min, expr)
        optimize!(model)
        st = termination_status(model)
        if st == MOI.OPTIMAL
            L[k] = objective_value(model)
        elseif st == MOI.DUAL_INFEASIBLE
            push!(unbounded_L, k)
        else
            push!(other_L, (k, string(st)))
        end
    end

    return L, U, unbounded_L, unbounded_U, other_L, other_U
end

# --------------------------------------------------------------------------
# parse a saved Gurobi log for diagnostics not exposed via MOI attributes
# --------------------------------------------------------------------------
function parse_gurobi_log(logtext::String)
    info = Dict{String,Any}(
        "presolved_rows"        => NaN,
        "presolved_cols"        => NaN,
        "bilinear_constraints"  => NaN,
        "root_relaxation_obj"   => NaN,
        "first_incumbent_found" => NaN,
        "root_gap"              => NaN,
    )

    m = match(r"Presolved:\s*(\d+)\s*rows,\s*(\d+)\s*columns", logtext)
    if m !== nothing
        info["presolved_rows"] = parse(Int, m.captures[1])
        info["presolved_cols"] = parse(Int, m.captures[2])
    end

    m = match(r"Presolved model has\s*(\d+)\s*bilinear constraint", logtext)
    if m !== nothing
        info["bilinear_constraints"] = parse(Int, m.captures[1])
    end

    m = match(r"Root relaxation: objective\s*([-+0-9.eE]+),", logtext)
    if m !== nothing
        info["root_relaxation_obj"] = parse(Float64, m.captures[1])
    end

    # first incumbent: either an explicit "Found heuristic solution" line,
    # or the first H-marked row in the node table, whichever appears first
    # in the log text.
    heur_m = match(r"Found heuristic solution: objective\s*([-+0-9.eE]+)", logtext)
    hrow_m = match(r"^H\s*\d+\s+\d+\s+.*?([-+0-9.eE]+)\s+[-+0-9.eE]+\s+[\d.]+%"m, logtext)

    heur_pos = heur_m === nothing ? typemax(Int) : heur_m.offset
    hrow_pos = hrow_m === nothing ? typemax(Int) : hrow_m.offset

    if heur_pos < typemax(Int) || hrow_pos < typemax(Int)
        if heur_pos <= hrow_pos
            info["first_incumbent_found"] = parse(Float64, heur_m.captures[1])
        else
            info["first_incumbent_found"] = parse(Float64, hrow_m.captures[1])
        end
    end

    if !isnan(info["root_relaxation_obj"]) && !isnan(info["first_incumbent_found"]) && info["first_incumbent_found"] != 0
        info["root_gap"] = abs(info["root_relaxation_obj"] - info["first_incumbent_found"]) / abs(info["first_incumbent_found"])
    end

    return info
end

# --------------------------------------------------------------------------
# NEW: y-space bounded reduced QP
# --------------------------------------------------------------------------
function solve_yspace_qp(A, B, c, Mvec, Lvec, Uvec, logfile; time_limit=300.0, mip_gap=1e-4)
    num_mu  = size(A, 1)
    n_delta = size(B, 2)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit", time_limit)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", mip_gap)
    set_optimizer_attribute(model, "LogFile", logfile)

    @variable(model, 0 <= mu[i=1:num_mu] <= Mvec[i])
    @variable(model, Lvec[k] <= y[k=1:n_delta] <= Uvec[k])
    @constraint(model, A' * mu .== 0.0)
    @constraint(model, dot(c, mu) == -1.0)
    @constraint(model, ydef, y .== B' * mu)
    @objective(model, Max, sum(y[k]^2 for k in 1:n_delta))

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
        y_star    = value.(y)
    else
        gamma_inc = NaN
        mu_star   = nothing
        y_star    = nothing
    end
    gamma_bd = try
        objective_bound(model)
    catch
        NaN
    end
    gap = isnan(gamma_inc) || gamma_inc == 0 ? NaN : abs(gamma_bd - gamma_inc) / abs(gamma_inc)

    return (status=status, runtime=runtime, nodes=nodes, gamma_inc=gamma_inc, gamma_bd=gamma_bd,
            gap=gap, mu=mu_star, y=y_star)
end

# --------------------------------------------------------------------------
# OLD-formulation replica, WITH Gurobi LogFile capture, solely so we can
# extract a comparable root-relaxation bound for the comparison table.
# This does not read or write reduced_qp_test.jl or its results; it is an
# independent re-solve of the identical old model, logged separately.
# --------------------------------------------------------------------------
function solve_muspace_qp_with_log(A, B, c, Mvec, logfile; time_limit=300.0, mip_gap=1e-4)
    num_mu = size(A, 1)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit", time_limit)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", mip_gap)
    set_optimizer_attribute(model, "LogFile", logfile)

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
    gamma_inc = has_values(model) ? objective_value(model) : NaN
    gamma_bd  = try
        objective_bound(model)
    catch
        NaN
    end
    gap = isnan(gamma_inc) || gamma_inc == 0 ? NaN : abs(gamma_bd - gamma_inc) / abs(gamma_inc)

    return (status=status, runtime=runtime, nodes=nodes, gamma_inc=gamma_inc, gamma_bd=gamma_bd, gap=gap)
end

# ==========================================================================
# per-case driver
# ==========================================================================
function run_case(casekey::String, time_limit::Float64; also_recompute_old_root_bound::Bool=true)
    case = CASE_MAP[casekey]
    println("="^78)
    println("Y-SPACE EXPERIMENT -- Case: $case  (key=$casekey, time_limit=$(time_limit)s)")
    println("="^78)

    network_data       = pglib(case)
    basic_network_data = PowerModels.make_basic_network(network_data)
    zero_nonlinear_costs!(basic_network_data)

    pm_result, model0, model_fl, sys = solve_dcopf(basic_network_data)
    A, B, c = copy(sys[:A]), copy(sys[:B]), copy(sys[:c])
    num_mu  = size(A, 1)
    n_delta = size(B, 2)

    # ---- mu-space bounds M[i] (same construction as reduced_qp_test.jl) ----
    mu_bound_time = @elapsed begin
        Mvec, unbounded_mu_idx, other_mu_idx = component_upper_bounds(A, c)
    end
    @printf("mu-bound preprocessing time = %.4f s (%d LPs)\n", mu_bound_time, num_mu)

    if !isempty(unbounded_mu_idx)
        println("STOP: mu-space components with no finite upper bound: $unbounded_mu_idx")
        return nothing
    end

    # ---- y-space bounds L[k], U[k], reusing one LP model ----
    y_bound_time = @elapsed begin
        Lvec, Uvec, unbounded_L, unbounded_U, other_L, other_U = compute_y_bounds(A, B, c)
    end
    @printf("y-bound preprocessing time  = %.4f s (%d LPs)\n", y_bound_time, 2 * n_delta)

    println()
    println("--- y-space LP bounds L[k] <= (B'mu)[k] <= U[k] ---")
    @printf("%-6s %-16s %-16s %-16s\n", "k", "L[k]", "U[k]", "U[k]-L[k]")
    for k in 1:n_delta
        Lk = isnan(Lvec[k]) ? "n/a" : @sprintf("%.8g", Lvec[k])
        Uk = isnan(Uvec[k]) ? "n/a" : @sprintf("%.8g", Uvec[k])
        wk = (isnan(Lvec[k]) || isnan(Uvec[k])) ? "n/a" : @sprintf("%.8g", Uvec[k] - Lvec[k])
        @printf("%-6d %-16s %-16s %-16s\n", k, Lk, Uk, wk)
    end

    if !isempty(unbounded_L) || !isempty(unbounded_U)
        println()
        println("STOP: y-space bounds are unbounded on this slice.")
        println("  unbounded L indices: $unbounded_L")
        println("  unbounded U indices: $unbounded_U")
        println("Nonlinear solve is skipped for this case so the geometry can be investigated.")
        return (casekey=casekey, stopped=true, reason="unbounded_y_bounds")
    end
    if !isempty(other_L) || !isempty(other_U)
        println()
        println("STOP: unexpected LP status while computing y-bounds.")
        println("  other L statuses: $other_L")
        println("  other U statuses: $other_U")
        return (casekey=casekey, stopped=true, reason="unexpected_lp_status")
    end

    minL   = minimum(Lvec)
    maxU   = maximum(Uvec)
    widths = Uvec .- Lvec
    maxW   = maximum(widths)
    avgW   = sum(widths) / length(widths)

    println()
    @printf("min_k L[k]        = %.8g\n", minL)
    @printf("max_k U[k]        = %.8g\n", maxU)
    @printf("max_k (U[k]-L[k]) = %.8g\n", maxW)
    @printf("avg_k (U[k]-L[k]) = %.8g\n", avgW)

    # ---- solve NEW y-space model ----
    yspace_logfile = joinpath(results_dir, "yspace_gurobi_$(casekey).log")
    qp_wall_time = @elapsed begin
        res_new = solve_yspace_qp(A, B, c, Mvec, Lvec, Uvec, yspace_logfile; time_limit=time_limit)
    end

    logtext_new  = read(yspace_logfile, String)
    parsed_new   = parse_gurobi_log(logtext_new)

    println()
    println("="^78)
    println("Y-SPACE RESULT: $case")
    println("="^78)
    @printf("termination status = %s\n", res_new.status)
    @printf("gamma_inc          = %.8g\n", res_new.gamma_inc)
    @printf("gamma_bd           = %.8g\n", res_new.gamma_bd)
    @printf("relative gap       = %.6g\n", res_new.gap)
    @printf("node count         = %.0f\n", res_new.nodes)
    @printf("solve runtime      = %.4f s\n", res_new.runtime)
    @printf("presolved rows/cols = %s / %s\n", parsed_new["presolved_rows"], parsed_new["presolved_cols"])
    @printf("bilinear constraints (presolved) = %s\n", parsed_new["bilinear_constraints"])
    @printf("root relaxation obj = %s\n", parsed_new["root_relaxation_obj"])
    @printf("first incumbent found (approx) = %s\n", parsed_new["first_incumbent_found"])
    @printf("approx root gap = %s\n", parsed_new["root_gap"])

    rho = (res_new.gamma_inc isa Number && !isnan(res_new.gamma_inc) && res_new.gamma_inc > 0) ? 1.0 / res_new.gamma_inc : NaN
    sqrt_rho = isnan(rho) ? NaN : sqrt(rho)
    @printf("rho = 1/gamma      = %.8g\n", rho)
    @printf("sqrt(rho)          = %.8g\n", sqrt_rho)

    # ---- equivalence / feasibility verification ----
    mu_star = res_new.mu
    y_star  = res_new.y
    max_AtMu   = NaN
    c_mu_p1    = NaN
    min_mu     = NaN
    max_ydef   = NaN
    obj_muform = NaN
    obj_yform  = NaN

    if mu_star !== nothing
        max_AtMu = maximum(abs.(A' * mu_star))
        c_mu_p1  = abs(dot(c, mu_star) + 1.0)
        min_mu   = minimum(mu_star)
        Bmu      = B' * mu_star
        max_ydef = maximum(abs.(y_star .- Bmu))
        obj_muform = dot(Bmu, Bmu)
        obj_yform  = sum(y_star .^ 2)

        println()
        println("--- equivalence / feasibility verification ---")
        @printf("max|A'mu*|              = %.6g\n", max_AtMu)
        @printf("|c'mu* + 1|             = %.6g\n", c_mu_p1)
        @printf("min_i mu_i*             = %.6g\n", min_mu)
        @printf("max_k |y_k* - (B'mu*)_k| = %.6g\n", max_ydef)
        @printf("sum_k (y_k*)^2          = %.8g\n", obj_yform)
        @printf("(mu*)'BB'mu*            = %.8g\n", obj_muform)
        @printf("difference               = %.6g\n", abs(obj_yform - obj_muform))
    else
        println()
        println("No incumbent returned -- cannot verify equivalence.")
    end

    # ---- (optional) old-formulation replica solve, logged, for root-bound comparison ----
    root_bound_old = NaN
    old_replica_status = "not_run"
    if also_recompute_old_root_bound
        old_logfile = joinpath(results_dir, "yspace_oldcompare_gurobi_$(casekey).log")
        res_old_replica = solve_muspace_qp_with_log(A, B, c, Mvec, old_logfile; time_limit=time_limit)
        logtext_old = read(old_logfile, String)
        parsed_old  = parse_gurobi_log(logtext_old)
        root_bound_old = parsed_old["root_relaxation_obj"]
        old_replica_status = res_old_replica.status
        println()
        println("--- old (mu-space) replica re-solve, for root-bound comparison only ---")
        @printf("old replica status  = %s\n", old_replica_status)
        @printf("old root relax obj  = %s\n", root_bound_old)
        @printf("old gamma_inc/gamma_bd (replica) = %.8g / %.8g\n", res_old_replica.gamma_inc, res_old_replica.gamma_bd)
    end

    # ---- persist per-case log ----
    logpath = joinpath(results_dir, "yspace_reduced_qp_$(casekey).txt")
    open(logpath, "w") do io
        println(io, "case = $case (key=$casekey)")
        println(io, "num_mu    = $num_mu")
        println(io, "n_delta   = $n_delta")
        println(io, "n_y_bound_LPs = $(2*n_delta)")
        @printf(io, "mu_bound_time = %.4f s\n", mu_bound_time)
        @printf(io, "y_bound_time  = %.4f s\n", y_bound_time)
        println(io)
        println(io, "--- y bounds ---")
        for k in 1:n_delta
            println(io, "  k=$k  L=$(Lvec[k])  U=$(Uvec[k])")
        end
        @printf(io, "min_k L[k]        = %.8g\n", minL)
        @printf(io, "max_k U[k]        = %.8g\n", maxU)
        @printf(io, "max_k (U[k]-L[k]) = %.8g\n", maxW)
        @printf(io, "avg_k (U[k]-L[k]) = %.8g\n", avgW)
        println(io)
        println(io, "--- Gurobi parsed diagnostics (new y-space model) ---")
        for (kk, vv) in parsed_new
            println(io, "  $kk = $vv")
        end
        println(io)
        println(io, "--- solve result (new y-space model) ---")
        @printf(io, "termination status = %s\n", res_new.status)
        @printf(io, "gamma_inc          = %.8g\n", res_new.gamma_inc)
        @printf(io, "gamma_bd           = %.8g\n", res_new.gamma_bd)
        @printf(io, "relative gap       = %.6g\n", res_new.gap)
        @printf(io, "node count         = %.0f\n", res_new.nodes)
        @printf(io, "solve runtime      = %.4f s\n", res_new.runtime)
        @printf(io, "qp wall time       = %.4f s\n", qp_wall_time)
        @printf(io, "rho = 1/gamma      = %.8g\n", rho)
        @printf(io, "sqrt(rho)          = %.8g\n", sqrt_rho)
        println(io)
        println(io, "--- equivalence / feasibility verification ---")
        @printf(io, "max|A'mu*|              = %.6g\n", max_AtMu)
        @printf(io, "|c'mu* + 1|             = %.6g\n", c_mu_p1)
        @printf(io, "min_i mu_i*             = %.6g\n", min_mu)
        @printf(io, "max_k |y_k* - (B'mu*)_k| = %.6g\n", max_ydef)
        @printf(io, "sum_k (y_k*)^2          = %.8g\n", obj_yform)
        @printf(io, "(mu*)'BB'mu*            = %.8g\n", obj_muform)
        if also_recompute_old_root_bound
            println(io)
            println(io, "--- old (mu-space) replica re-solve (root-bound comparison only) ---")
            @printf(io, "old replica status  = %s\n", old_replica_status)
            @printf(io, "old root relax obj  = %s\n", root_bound_old)
        end
    end

    total_runtime = mu_bound_time + y_bound_time + qp_wall_time

    csvpath = joinpath(results_dir, "yspace_reduced_qp_summary.csv")
    header_needed = !isfile(csvpath)
    open(csvpath, "a") do cio
        if header_needed
            println(cio, join([
                "case","num_mu","n_delta","n_y_bound_lps","mu_bound_time_s","y_bound_time_s",
                "min_L","max_U","max_width","avg_width",
                "presolved_rows","presolved_cols","bilinear_constraints",
                "root_relaxation_obj","first_incumbent_found","approx_root_gap",
                "gamma_inc","gamma_bd","rel_gap","nodes","solve_runtime_s","total_runtime_s",
                "termination_status","rho","sqrt_rho",
                "max_AtMu","c_mu_plus1","min_mu","max_ydef","obj_yform","obj_muform",
                "root_bound_old_replica","old_replica_status"
            ], ","))
        end
        println(cio, join([
            casekey, num_mu, n_delta, 2*n_delta, mu_bound_time, y_bound_time,
            minL, maxU, maxW, avgW,
            parsed_new["presolved_rows"], parsed_new["presolved_cols"], parsed_new["bilinear_constraints"],
            parsed_new["root_relaxation_obj"], parsed_new["first_incumbent_found"], parsed_new["root_gap"],
            res_new.gamma_inc, res_new.gamma_bd, res_new.gap, res_new.nodes, res_new.runtime, total_runtime,
            res_new.status, rho, sqrt_rho,
            max_AtMu, c_mu_p1, min_mu, max_ydef, obj_yform, obj_muform,
            root_bound_old, old_replica_status
        ], ","))
    end

    println()
    println("Saved: $logpath")
    println("Appended: $csvpath")
    println("Gurobi log (new):  $yspace_logfile")
    if also_recompute_old_root_bound
        println("Gurobi log (old replica): $(joinpath(results_dir, "yspace_oldcompare_gurobi_$(casekey).log"))")
    end

    return (casekey=casekey, stopped=false)
end

# ==========================================================================
# main: casekey / time_limit from CLI, or "all" to run all seven in order
# ==========================================================================
casekey_arg = length(ARGS) >= 1 ? ARGS[1] : "5_pjm"
time_limit  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 300.0

if casekey_arg == "all"
    for ck in ALL_CASES
        run_case(ck, time_limit)
    end
else
    run_case(casekey_arg, time_limit)
end
