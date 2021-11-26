# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using Revise
using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit
using LinearAlgebra

A = zeros(ComplexF64, (2,2,2,2))
A[1,1,1,1] = A[2,2,2,2] = 0.5
A[1,2,1,2] = A[2,1,2,1] = 1

reshape(permutedims(A, (1,3,2,4)), (4,4)) |> svd





####################################################################
# MPO for the triangular AF Ising
T = mpo_triangular_AF_ising()
Tb = mpo_triangular_AF_ising_alternative()
T_vonb, T_tob = mpo_triangular_AF_ising_adapter()

psi = quickload("ckpt_variational_chi$(chi)")
_, psi = left_canonical(psi)
_, s, _ = toarray(psi)[:, 1, :] ^ 10 |> svd; s

####################################################################
space(T_vonb)
T_vonb_arr = reshape(T_vonb.data, (4,2,4,4))
space(T_tob) 
T_tob_arr = reshape(T_tob.data, (2,4,2,2))

psi = quickload("ckpt_iTEBD_chi32")
_, s, _ = toarray(psi)[:, 1, :] ^ 10 |> svd; s
_, s, _ = (toarray(psi)[:, 3, :] * toarray(psi)[:, 2, :])^5 |> svd; s

M1 = zeros(ComplexF64, (3, 3))
M1[1,1] = 1;
M1[2,2] = M1[3,2] = 0.5;
M1[3,3] = 1;
M1

M2 = zeros(ComplexF64, (3, 3))
M2[1,1] = 1;
M2[2,2] = 0.5;
M2[1,2] = M2[3,2] = 0.25;
M2[3,3] = 0.75;
M2[2,3] = 0.25;
M2

eigvals(M1)
eigvals(M2)

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