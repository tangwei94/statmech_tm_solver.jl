# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

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

ψ = cmps(rand, 4, 2)



c = 1
μ = 1
L = 1000
function f(arr::Array{ComplexF64, 3})
    ψ = convert_to_cmps(arr)
    return energy_lieb_liniger(ψ, c, L, μ)
end

function g!(garr::Array{ComplexF64, 3}, arr::Array{ComplexF64, 3})
    garr .= f'(arr)
end

res = optimize(f, g!, ψ_arr, LBFGS(), Optim.Options(show_trace=true, iterations=100))
ψm = convert_to_cmps(Optim.minimizer(res));

E = energy_lieb_liniger(ψm, c, L, μ)
env = finite_env(K_mat(ψm, ψm), L)
env = permute(env, (2, 3), (4, 1))

op_ρ = particle_density(ψm)
tr(env * op_ρ)