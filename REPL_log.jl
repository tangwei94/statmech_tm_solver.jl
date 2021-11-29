# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using TensorKit
using TensorOperations
using KrylovKit
using LinearAlgebra
using Revise
using statmech_tm_solver

####################################################################
M = rand(ComplexF64, (3,3))
triu(M, 1) + 0.5 * Diagonal(M)

####################################################################
# MPO for the triangular AF Ising
T = mpo_triangular_AF_ising()
Tb = mpo_triangular_AF_ising_alternative()
T_vonb, T_tob = mpo_triangular_AF_ising_adapter()

psi = quickload("ckpt_variational_chi$(chi)")
_, psi = left_canonical(psi)

exp(-Inf)
Tn = mpo_kink_processor(Inf)
_, Tn_psi = left_canonical(act(Tn, psi))

ovlp(psi, Tn_psi)

