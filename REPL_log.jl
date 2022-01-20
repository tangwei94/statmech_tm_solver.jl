# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using TensorKit
using TensorOperations
using TensorKitAD
using Zygote
using KrylovKit
using LinearAlgebra
using Plots
using Optim

using Revise
using statmech_tm_solver

chis = [4; 8; 12; 16; 20; 24; 28; 32]
beta = 4
for (ix, chi) in enumerate(chis)
    ψ_data = quickload("ckpt_cMPO_ising_beta$(beta)_chi$(chi)")
    ψ = convert_to_cmps(ψ_data)
    K = K_mat(ψ, ψ)

    K = 0.5*(K+K')

    W, _ = eigh(K)
    Wvec = diag(W.data)
    @show get_chi(ψ), -beta*(Wvec .- Wvec[end])[end-5:end]
end

T = cmpo_ising(1.0)
ϕ = cmps(T.Q, T.R)
for ix in 1:3
    ϕ = act(T, ϕ)
end
get_chi(ϕ)
_, _, ϕ1 = compress(ϕ, 6, 8)
truncation_check(ϕ, ϕ1, 8)


qchi = 16
ψ_data = quickload("ckpt_cMPO_ising_beta32_chi$(chi)")
ψ = convert_to_cmps(ψ_data)
K = K_mat(ψ, ψ)
K = 0.5*(K+K')

W, _ = eigh(K)
Wvec = diag(W.data)
@show -(Wvec[end-1] - Wvec[end]) * 32 - sqrt(2)