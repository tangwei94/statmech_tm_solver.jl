# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using Revise
using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit

chi = 8
psi = qbimps(rand, chi, 2, 1)
Γ = 1.

T = cmpo_ising(Γ)
energy_quantum_ising(psi.A, Γ)
println("========================================================================")

for ix in 1:100
    psi = A_canonical(T, psi)
    psi = B_canonical(T, psi)
    println(energy_quantum_ising(psi.A, Γ)|>real, ' ', -4/pi)
end

sA = entanglement_spectrum(psi.A)
sB = entanglement_spectrum(psi.B)