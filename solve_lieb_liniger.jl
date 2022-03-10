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
nsteps = 100

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
    P1 = preconditioner(ψ, L)
    P.map = P1.map
    P.invmap = P1.invmap
    P.proj = P1.proj
end

ψms, Es = [], []
ψm = cmps(rand, 2, d)

for χ in [2; 4; 8; 12; 16]
    global ψm

    #ψ0 = expand(ψm, χ, 0.01)
    ψ0 = quickload("lieb_liniger_c$(c)_mu$(μ)_L$(L)_chi$(χ)") |> convert_to_cmps
    ψ_arr0 = convert_to_array(ψ0)
    E0 = energy_lieb_liniger(ψ0, c, L, μ)

    res = optimize(f, g!, ψ_arr0, LBFGS(), Optim.Options(show_trace=true, iterations=nsteps))
    ψ_arr = Optim.minimizer(res);
    ψm = convert_to_cmps(ψ_arr);

    P0 = preconditioner(ψm, L);
    res = optimize(f, g!, ψ_arr, LBFGS(P = P0, precondprep=_precondprep!), Optim.Options(show_trace=true, iterations=2*nsteps))
    ψm = convert_to_cmps(Optim.minimizer(res));

    # calculate observables. total, kinetic and potential energies
    E = energy_lieb_liniger(ψm, c, L, μ)
    env = finite_env(K_mat(ψm, ψm), L)
    env = permute(env, (2, 3), (4, 1))

    op_ρ = particle_density(ψm)
    op_k = kinetic(ψm)
    ρ_value = tr(env * op_ρ)
    k_value = tr(env * op_k)

    push!(ψms, ψm)
    push!(Es, E)
    @show χ, E, ρ_value, k_value
    
    quicksave("lieb_liniger_c$(c)_mu$(μ)_L$(L)_chi$(χ)", convert_to_tensormap(ψm))
end

#########################################################
ψm = quickload("lieb_liniger_c$(c)_mu$(μ)_L$(L)_chi8") |> convert_to_cmps
ψm = normalize(ψm, L, sym=false)
E = energy_lieb_liniger(ψm, c, L, μ) 

env = finite_env(K_mat(ψm, ψm), L)
env = permute(env, (2, 3), (4, 1))
op_ρ = particle_density(ψm)
op_k = kinetic(ψm)
ρ_value = tr(env * op_ρ)
k_value = tr(env * op_k)
γ = c / ρ_value

shift = 0
#for ix in -20:20
    ix = 0
    p = ix * 2 * pi / L
    χ = get_chi(ψm)
    V0 = TensorMap(rand, ComplexF64, ℂ^(χ*d), ℂ^χ)
    h_lop = lieb_liniger_h_tangent_map(ψm, p, 0, c, L, μ)
    n_lop = tangent_map(ψm, L, p)

    Wn, Vn = eigsolve(n_lop, V0, d*χ^2; tol=1e-8, krylovdim=d*χ^2, ishermitian=true)
    invWn = zero(Wn)
    invWn[Wn .> δ] = 1 ./ Wn[Wn .> δ]
    invWn[Wn .<= δ] .= 0
    #@benchmark h_lop(V0)
    #@benchmark n_lop(V0)
    #@benchmark linsolve(V -> n_lop(V), V0, V0; krylovdim=χ^2, ishermitian=true, isposdef=true)

    function linsolve_with_eig(V0::TensorMap{ComplexSpace, 1, 1})
        tmpf = (Vx, invWx) -> Vx * tr(Vx' * V0) * invWx#/ (Wx + δ)
        return sum(tmpf.(Vn, invWn)) 
    end
    #@benchmark linsolve_with_eig(V0)

    #@show geneigsolve(v->(h_lop(v)/L, n_lop(v)), V0, 2*χ, :SR ; ishermitian=true, isposdef=true)[1]
    Ee = eigsolve(h_lop ∘ linsolve_with_eig, V0, 1, :SR)[1] ./ L
    @show ix, Ee 
#end
