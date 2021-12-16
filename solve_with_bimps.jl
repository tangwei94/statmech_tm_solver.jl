using TensorKit
using statmech_tm_solver

#log(1+sqrt(2))/2
#T = mpo_square_ising(log(1+sqrt(2))/2)
T = mpo_triangular_AF_ising_alternative()
#T = mpo_triangular_AF_ising()

psi = bimps(rand, 12, 4)
for ix in 1:50
    global psi
    λA, psi = A_canonical(T, psi)
    λB, psi = B_canonical(T, psi)
    print("$ix, $λA  $λB ")
    println(' ', free_energy(T, psi.B), ' ', free_energy(T, psi.A))
end