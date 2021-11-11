# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using Revise
using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit

T = mpo_square_ising(0.5)
psi = bimps(rand, 8, 2)

for ix in 1:30
    psi = B_canonical(T, psi);
    psi = A_canonical(T, psi);
    println(free_energy(T, psi.A))
end