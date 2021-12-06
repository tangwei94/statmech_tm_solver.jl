using Revise
using statmech_tm_solver
using TensorKit
using Zygote
using LinearAlgebra
using Optim

# MPO 
#T = mpo_triangular_AF_ising()
T = mpo_triangular_AF_ising_alternative()
T2 = mpo_triangular_AF_ising()
T_adapter, _ = mpo_triangular_AF_ising_adapter()
d = 4

psi = TensorMap(rand, ComplexF64, ℂ^2*ℂ^d, ℂ^2)
_, psi = left_canonical(psi)

io = open("result_iTEBD.txt", "w")
close(io)
for chi in [2; 4; 8; 16; 32; 64]
    global psi
    convergence_flag = -10
    io = open("result_iTEBD.txt", "a+")
    while convergence_flag <= 0
        _, Tpsi = left_canonical(act(T, psi))
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

        msg = "$chi $F_value $costfunc_value $fidelity_value $F_value2 $costfunc_value2"

        println(msg)
        println(io, msg)
    end
    close(io)
    quicksave("ckpt_iTEBD_chi$(chi)", psi)
end