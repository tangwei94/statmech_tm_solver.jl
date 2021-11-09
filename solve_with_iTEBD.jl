using Revise
using statmech_tm_solver
using TensorKit
using Zygote
using LinearAlgebra
using Optim

# MPO 
#T = mpo_triangular_AF_ising()
beta_c = log(1 + sqrt(2)) / 2
T = mpo_square_ising(beta_c)

psi = TensorMap(rand, ComplexF64, ℂ^2*ℂ^2, ℂ^2)
_, psi = left_canonical(psi)

io = open("result.txt", "w+")
for chi in [2; 4; 8; 16]
    for ix in 1:50
        _, Tpsi = left_canonical(act(T, psi))
        _, psi = iTEBD_truncate(Tpsi, chi)
        #psi = variational_truncate(Tpsi, chi)
        println(chi, ' ', free_energy(T, psi), ' ', nonherm_cost_func(T, toarray(psi)),' ', ln_fidelity(psi, Tpsi))
        println(io, chi, ' ', free_energy(T, psi), ' ', nonherm_cost_func(T, toarray(psi)), ' ', ln_fidelity(psi, Tpsi))
    end
    
    _, psi = left_canonical(act(T, psi)) 

end

close(io)
