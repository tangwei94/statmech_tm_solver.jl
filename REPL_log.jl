# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using Revise
using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit

psi = qbimps(rand, 6, 2, 1)
T = cmpo_ising(0.0)
energy_quantum_ising(psi.A, 0.0)

psi = B_canonical(T, psi)
psi = A_canonical(T, psi)
energy_quantum_ising(psi.A, 0.0)

for ix in 1:100
    psi = B_canonical(T, psi)
    psi = A_canonical(T, psi)
    println(energy_quantum_ising(psi.A, 0.0))
end
