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
using Suppressor

using Revise
using statmech_tm_solver

#########################################################

# parameters 
c = 1
μ = 1
L = 50 
d = 1
nsteps1 = 10 # ordinary LBFGS steps
nsteps2 = 200 # preconditioned LBFGS steps
precond_tol = 1e-5 # preconditioner = inv(g + precond_tol * norm(g) * Id)

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
    P1 = preconditioner(ψ, L; tol=1e-5)
    P.map = P1.map
    P.invmap = P1.invmap
    P.proj = P1.proj
end

ψm = cmps(rand, 2, d)
#ψm = quickload("lieb_liniger_c$(c)_mu$(μ)_L$(L)_chi4") |> convert_to_cmps

for χ in [2; 4; 8; 12; 16; 20]
    global ψm

    #ψ0 = expand(ψm, χ, 0.01)
    ψ0 = quickload("lieb_liniger_c$(c)_mu$(μ)_L$(L)_chi$(χ)") |> convert_to_cmps

    ψ_arr0 = convert_to_array(ψ0)
    E0 = energy_lieb_liniger(ψ0, c, L, μ)

    res = optimize(f, g!, ψ_arr0, LBFGS(), Optim.Options(show_trace=true, iterations=nsteps1))
    ψ_arr = Optim.minimizer(res);
    ψm = convert_to_cmps(ψ_arr);

    P0 = preconditioner(ψm, L);
    res = optimize(f, g!, ψ_arr, LBFGS(P = P0, precondprep=_precondprep!), Optim.Options(show_trace=true, iterations=nsteps2))
    ψm = convert_to_cmps(Optim.minimizer(res));

    # calculate observables. total, kinetic and potential energies
    E = energy_lieb_liniger(ψm, c, L, μ)
    env = finite_env(K_mat(ψm, ψm), L)
    env = permute(env, (2, 3), (4, 1))

    op_ρ = particle_density(ψm)
    op_k = kinetic(ψm)
    ρ_value = tr(env * op_ρ)
    k_value = tr(env * op_k)

    @show χ, E, ρ_value, k_value
    
    quicksave("lieb_liniger_c$(c)_mu$(μ)_L$(L)_chi$(χ)", convert_to_tensormap(ψm))
end

