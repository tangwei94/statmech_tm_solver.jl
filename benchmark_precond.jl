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

#ψ0 = cmps(rand, χ, 1)
ψ0 = quickload("lieb_liniger_c$(c)_mu$(μ)_L$(L)_chi$(χ-4)") |> convert_to_cmps
ψ0 = expand(ψ0, χ, 0.001)
ψ_arr0 = convert_to_array(ψ0)
E0 = energy_lieb_liniger(ψ0, c, L, μ)

res = optimize(f, g!, ψ_arr0, LBFGS(), Optim.Options(show_trace=true, iterations=nsteps))
ψ_arr = Optim.minimizer(res);
ψm = convert_to_cmps(ψ_arr);
ψm_grad = convert_to_tensormap(f'(ψ_arr), 2)
E1 = energy_lieb_liniger(ψm, c, L, μ)

# optimize with preconditioners
function _precondprep!(P::preconditioner, ψ_arr::Array{ComplexF64, 3})
    ψ = convert_to_cmps(ψ_arr)
    ψgrad = convert_to_tensormap(f'(ψ_arr), 2)

    P1 = preconditioner(ψ, ψgrad, L; gauge=:periodic)
    P.map = P1.map
    P.invmap = P1.invmap
    P.proj = P1.proj
end

P0 = preconditioner(ψm, ψm_grad, L);
res_w_precond = optimize(f, g!, ψ_arr, LBFGS(P = P0, precondprep=_precondprep!), Optim.Options(show_trace=true, store_trace=true, iterations=nsteps))
ψm_w_precond = convert_to_cmps(Optim.minimizer(res_w_precond));

trace_w_precond = Optim.trace(res_w_precond)
times_w_precond = [trace_w_precond[ix].metadata["time"] for ix in 1:nsteps+1]
values_w_precond = [trace_w_precond[ix].value for ix in 1:nsteps+1]
gnorms_w_precond = [trace_w_precond[ix].g_norm for ix in 1:nsteps+1]

# optimize without preconditioners
res_wo_precond = optimize(f, g!, ψ_arr, LBFGS(), Optim.Options(show_trace=true, store_trace=true, iterations=2*nsteps))
ψm_wo_precond = convert_to_cmps(Optim.minimizer(res_wo_precond));

trace_wo_precond = Optim.trace(res_wo_precond)
times_wo_precond = [trace_wo_precond[ix].metadata["time"] for ix in 1:2*nsteps+1]
values_wo_precond = [trace_wo_precond[ix].value for ix in 1:2*nsteps+1]
gnorms_wo_precond = [trace_wo_precond[ix].g_norm for ix in 1:2*nsteps+1]

plot(times_w_precond, log10.(gnorms_w_precond))
plot!(times_wo_precond, log10.(gnorms_wo_precond))

plot(times_w_precond, values_w_precond)
plot!(times_w_precond, values_w_precond); @show "ploted"
plot!(times_wo_precond, values_wo_precond)