# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using Revise
using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit

#log(1+sqrt(2))/2
#T = mpo_square_ising(log(1+sqrt(2))/2)
T = mpo_triangular_AF_ising_alternative()
psi = bimps(rand, 12, 4)

for ix in 1:50
    psi = A_canonical(T, psi)
    psi = B_canonical(T, psi)
    println(ix, ' ', free_energy(T, psi.B), ' ', free_energy(T, psi.A))
end