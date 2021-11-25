# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using Revise
using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit
using LinearAlgebra

####################################################################
# MPO for the triangular AF Ising
T = mpo_triangular_AF_ising()
Tb = mpo_triangular_AF_ising_alternative()
T_vonb, T_tob = mpo_triangular_AF_ising_adapter()

chi = 32
psi = quickload("ckpt_variational_chi$(chi)")
_, psi = iTEBD_truncate(act(T_tob, psi), chi)

nonherm_cost_func(Tb, psi)

####################################################################

identity = Matrix{ComplexF64}(I, 8, 8)
AQ = rand(ComplexF64, 8, 8)
AR = rand(ComplexF64, 8, 2, 8)
ϵ = 1e-12

A = zeros(ComplexF64, 8, 3, 8)
A0 = zeros(ComplexF64, 8, 3, 8)
A0[:, 1, :] = identity
A[:, 1, :] = identity + ϵ * AQ
A[:, 2:end, :] = sqrt(ϵ) * AR

Q, R = qr(reshape(A, (24, 8)))

tildeR = 0.5 * (AQ + AQ')

Ys = Array(1:8)
Xs = Array(transpose(1:8))
Xs = repeat(Xs, 8, 1)
Ys = repeat(Ys, 1, 8) 

tildeR[Xs .< Ys] .= 0
tildeR -= 0.5 * Diagonal(tildeR)

tildeQQ = AQ - tildeR 
tildeQR = AR

Q1 = zeros(ComplexF64, 8, 3, 8)
Q1[:, 1, :] = identity + ϵ * tildeQQ
Q1[:, 2:end, :] = sqrt(ϵ) * tildeQR

R1 = identity + ϵ * tildeR

using OMEinsum

println(ein"abc,cd->abd"(Q1, R1) ≈ A)

println(ein"abc,abd->cd"(conj(Q1), Q1) ≈ identity)