# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using Revise
using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit

psi = qbimps(rand, 2, 2, 1)
Γ=0.7025

T = cmpo_ising(Γ)
energy_quantum_ising(psi.A, Γ)

psi = B_canonical(T, psi)
energy_quantum_ising(psi.A, Γ)
psi = A_canonical(T, psi)
energy_quantum_ising(psi.A, Γ)

for ix in 1:100
    psi = B_canonical(T, psi)
    psi = A_canonical(T, psi)
    println(energy_quantum_ising(psi.A, Γ))
end
