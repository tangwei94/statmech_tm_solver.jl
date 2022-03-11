# test the performance of preconditioner

using TensorKit
using TensorOperations
using TensorKitAD
using KrylovKit
using Zygote
using KrylovKit
using LinearAlgebra
using Plots
using Optim
using ChainRules
using ChainRulesCore

using Revise
using statmech_tm_solver

#########################################################

# parameters 
c = 1
μ = 1
L = 50 
d = 1
nsteps = 200
χ = 12

# cost function (energy) and its gradient
function f(ψ_arr::Array{ComplexF64, 3})
    ψ = convert_to_cmps(ψ_arr)
    return energy_lieb_liniger(ψ, c, L, μ)
end

function g!(ψg_arr::Array{ComplexF64, 3}, ψ_arr::Array{ComplexF64, 3})
    copyto!(ψg_arr, f'(ψ_arr))
end

# construct the preconditioner
function _precondprep!(P::preconditioner, ψ_arr::Array{ComplexF64, 3})
    ψ = convert_to_cmps(ψ_arr)
    P1 = preconditioner(ψ, L; gauge=:periodic)
    P.map = P1.map
    P.invmap = P1.invmap
    P.proj = P1.proj
end
function _precondprep1!(P::preconditioner, ψ_arr::Array{ComplexF64, 3})
    ψ = convert_to_cmps(ψ_arr)
    proj = gauge_fixing_proj(ψ, L)
    tol = norm(proj' * convert_to_tensormap(f'(ψ_arr), 2))
    P1 = preconditioner(ψ, L; tol=tol)
    P.map = P1.map
    P.invmap = P1.invmap
    P.proj = P1.proj
end

#ψ0 = cmps(rand, χ, 1)
ψ0 = quickload("lieb_liniger_c$(c)_mu$(μ)_L$(L)_chi$(χ-4)") |> convert_to_cmps
ψ0 = expand(ψ0, χ, 0.01)
ψ_arr0 = convert_to_array(ψ0)
E0 = energy_lieb_liniger(ψ0, c, L, μ)

res = optimize(f, g!, ψ_arr0, LBFGS(), Optim.Options(show_trace=true, iterations=nsteps))
ψ_arr = Optim.minimizer(res);
ψm = convert_to_cmps(ψ_arr);
E1 = energy_lieb_liniger(ψm, c, L, μ)

# optimize with preconditioners
P0 = preconditioner(ψm, L);
res_w_precond = optimize(f, g!, ψ_arr, LBFGS(P = P0, precondprep=_precondprep!), Optim.Options(show_trace=true, store_trace=true, iterations=nsteps))
ψm_w_precond = convert_to_cmps(Optim.minimizer(res_w_precond));

trace_w_precond = Optim.trace(res_w_precond)
times_w_precond = [trace_w_precond[ix].metadata["time"] for ix in 1:nsteps+1]
values_w_precond = [trace_w_precond[ix].value for ix in 1:nsteps+1]
gnorms_w_precond = [trace_w_precond[ix].g_norm for ix in 1:nsteps+1]

# optimize with preconditioners, test another set of parameter in the preconditioner
P0 = preconditioner(ψm, L);
res_w_precond1 = optimize(f, g!, ψ_arr, LBFGS(P = P0, precondprep=_precondprep1!), Optim.Options(show_trace=true, store_trace=true, iterations=nsteps))
ψm_w_precond1 = convert_to_cmps(Optim.minimizer(res_w_precond1));

trace_w_precond1 = Optim.trace(res_w_precond1)
times_w_precond1 = [trace_w_precond1[ix].metadata["time"] for ix in 1:nsteps+1]
values_w_precond1 = [trace_w_precond1[ix].value for ix in 1:nsteps+1]
gnorms_w_precond1 = [trace_w_precond1[ix].g_norm for ix in 1:nsteps+1]

# optimize without preconditioners
#res_wo_precond = optimize(f, g!, ψ_arr, LBFGS(), Optim.Options(show_trace=true, store_trace=true, iterations=2*nsteps))
#ψm_wo_precond = convert_to_cmps(Optim.minimizer(res_wo_precond));
#
#trace_wo_precond = Optim.trace(res_wo_precond)
#times_wo_precond = [trace_wo_precond[ix].metadata["time"] for ix in 1:2*nsteps+1]
#values_wo_precond = [trace_wo_precond[ix].value for ix in 1:2*nsteps+1]
#gnorms_wo_precond = [trace_wo_precond[ix].g_norm for ix in 1:2*nsteps+1]

plot(times_w_precond, log10.(gnorms_w_precond))
plot!(times_w_precond1, log10.(gnorms_w_precond1))
#plot!(times_wo_precond, log10.(gnorms_wo_precond))

plot(1:nsteps+1, log10.(gnorms_w_precond))
plot!(1:nsteps+1, log10.(gnorms_w_precond1))

msk = (nsteps÷2):nsteps+1 
plot(times_w_precond[msk], values_w_precond[msk])
plot!(times_w_precond1[msk], values_w_precond1[msk])
#plot!(times_wo_precond, values_wo_precond)

plot((nsteps÷2):nsteps+1, values_w_precond[msk])
plot!((nsteps÷2):nsteps+1, values_w_precond1[msk])