# solve the ground state of Lieb-Liniger model
# also demonstrates the usage of preconditioners in Optim.jl

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

# parameter for lieb-liniger 
c = 1
μ = 1
L = 50 

# bond dimensions
d, χ = 1, 8

# random initial guess
ψ_arr = rand(ComplexF64, (χ, d+1, χ))
convert_to_cmps(ψ_arr)

# cost function (energy) and its gradient
function f(ψ_arr::Array{ComplexF64, 3})
    ψ = convert_to_cmps(ψ_arr)
    return energy_lieb_liniger(ψ, c, L, μ)
end

function g!(ψg_arr::Array{ComplexF64, 3}, ψ_arr::Array{ComplexF64, 3})
    copyto!(ψg_arr, f'(ψ_arr))
end

# use LBFGS to optimize the energy for 250 steps
res = optimize(f, g!, ψ_arr, LBFGS(), Optim.Options(show_trace=true, iterations=250))
ψm = convert_to_cmps(Optim.minimizer(res));
ψm_arr = convert_to_array(ψm);

# calculate observables. total, kinetic and potential energies
E = energy_lieb_liniger(ψm, c, L, μ)
env = finite_env(K_mat(ψm, ψm), L)
env = permute(env, (2, 3), (4, 1))

op_ρ = particle_density(ψm)
op_k = kinetic(ψm)
tr(env * op_ρ)
tr(env * op_k)

#########################################################
# now we test the performance of the preconditioner

# construct the preconditioner
function _precondprep!(P::preconditioner, ψ_arr::Array{ComplexF64, 3})
    ψ = convert_to_cmps(ψ_arr)
    P.map = tangent_map(ψ, L)
    P.proj = gauge_fixing_proj(ψ, L)
end

# use preconditioned LBFGS, optimize the energy for additional 50 steps
P0 = preconditioner(ψm, L)
res1 = optimize(f, g!, ψm_arr, LBFGS(P = P0, precondprep=_precondprep!), Optim.Options(show_trace=true, iterations=50))
ψm1 = convert_to_cmps(Optim.minimizer(res1));
E1 = energy_lieb_liniger(ψm1, c, L, μ)
@show E1 - E

# use ordinary LBFGS, optimize the energy for additional 50 steps
res2 = optimize(f, g!, ψm_arr, LBFGS(), Optim.Options(show_trace=true, iterations=50))
ψm2 = convert_to_cmps(Optim.minimizer(res2));
E2 = energy_lieb_liniger(ψm2, c, L, μ)
@show E2 - E

# from original random init guess, 100 steps of ordinary LBFGS + 100 steps of preconditioned GradientDescent + 100 steps of preconditioned LBFGS
res3 = optimize(f, g!, ψ_arr, LBFGS(), Optim.Options(show_trace=true, iterations=100))
ψ3_arr = Optim.minimizer(res3)
ψm3 = convert_to_cmps(ψ3_arr)

P0 = preconditioner(ψm3, L)
res3 = optimize(f, g!, ψ3_arr, GradientDescent(P = P0, precondprep=_precondprep!), Optim.Options(show_trace=true, iterations=100))
ψ3_arr = Optim.minimizer(res3)
ψm3 = convert_to_cmps(ψ3_arr)

P0 = preconditioner(ψm3, L)
res3 = optimize(f, g!, ψ3_arr, LBFGS(P = P0, precondprep=_precondprep!), Optim.Options(show_trace=true, iterations=100))
ψm3 = convert_to_cmps(Optim.minimizer(res3));

E3 = energy_lieb_liniger(ψm3, c, L, μ)
@show E3 - E