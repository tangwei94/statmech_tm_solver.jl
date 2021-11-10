using Revise
using statmech_tm_solver
using TensorKit
using Zygote
using LinearAlgebra
using Optim

# MPO 
#T = mpo_triangular_AF_ising()
T = mpo_triangular_AF_ising_alternative()
d = 4

psi = TensorMap(rand, ComplexF64, ℂ^2*ℂ^d, ℂ^2)
_, psi = left_canonical(psi)

io = open("result.txt", "w")
close(io)
for chi in [2; 4; 8; 16; 32]
    convergence_flag = -10
    io = open("result.txt", "a+")
    while convergence_flag <= 0
        _, Tpsi = left_canonical(act(T, psi))
        _, psi = iTEBD_truncate(Tpsi, chi)
        #psi = variational_truncate(Tpsi, chi)

        F_value_prev = F_value
        F_value = free_energy(T, psi)
        costfunc_value = nonherm_cost_func(T, toarray(psi))
        fidelity_value = ln_fidelity(psi, Tpsi)

        if isapprox(costfunc_value, -fidelity_value; rtol=0.02, atol=1e-13)
            convergence_flag += 1
        end

        println(chi, ' ', F_value, ' ', costfunc_value, ' ', fidelity_value)
        println(io, chi, ' ', F_value, ' ', costfunc_value, ' ', fidelity_value)
    end
    close(io)
    
    _, Tpsi = left_canonical(act(T, psi))
end

