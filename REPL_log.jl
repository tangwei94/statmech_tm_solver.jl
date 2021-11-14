# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using Revise
using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit

psi = qbimps(rand, 6, 2, 1)
T = cmpo_ising(1.0)
TA = mpo_quantum_ising(1.0)
psi = B_canonical(T, psi)
psi = A_canonical(T, psi)
println(exp(free_energy(TA, psi.A)))

for ix in 1:10
    psi = B_canonical(T, psi)
    psi = A_canonical(T, psi)
    println(exp(free_energy(TA, psi.A)))
end
