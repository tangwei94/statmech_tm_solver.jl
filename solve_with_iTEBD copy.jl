using Revise
using statmech_tm_solver
using TensorKit
using Zygote
using LinearAlgebra
using Optim

# MPO 
#T = mpo_triangular_AF_ising()
T2 = mpo_triangular_AF_ising_alternative()
T = mpo_triangular_AF_ising()
_, T_adapter = mpo_triangular_AF_ising_adapter()
d = 2

psi = TensorMap(rand, ComplexF64, ℂ^2*ℂ^d, ℂ^2)
_, psi = left_canonical(psi)

io = open("result_iTEBD1.txt", "w")
close(io)
for chi in [2; 4; 8; 16; 32]
    convergence_flag = -10
    io = open("result_iTEBD1.txt", "a+")
    ix = 0
    while convergence_flag <= 0 && ix < 100
        ix +=1

        _, Tpsi = left_canonical(act(T,act(T, psi)))
        _, psi = iTEBD_truncate(Tpsi, chi)
        #psi = variational_truncate(Tpsi, chi)

        F_value = free_energy(T, psi)
        F_value2 = free_energy(T2, act(T_adapter, psi))
        costfunc_value = nonherm_cost_func(T, psi)
        costfunc_value2 = nonherm_cost_func(T2, act(T_adapter, psi))
        fidelity_value = ln_fidelity(psi, Tpsi)

        if isapprox(costfunc_value, -fidelity_value; rtol=0.02, atol=1e-13)
            convergence_flag += 1
        end

        msg = "$chi $F_value $F_value2 $costfunc_value $fidelity_value $costfunc_value2"

        println(msg)
        println(io, msg)

        quicksave("ckpt_iTEBD1_chi$(chi)", psi)

    end
    close(io)
end