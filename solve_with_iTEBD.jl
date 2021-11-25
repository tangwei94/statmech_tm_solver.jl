using Revise
using statmech_tm_solver
using TensorKit
using Zygote
using LinearAlgebra
using Optim

function expct(T::TensorMap{ComplexSpace, 2, 2}, psi::TensorMap{ComplexSpace, 2, 1})
    Tpsi = act(T, psi)
    return ovlp(psi, Tpsi) / ovlp(psi, psi)
end

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
for chi in [2; 4; 8; 16; 32]
    convergence_flag = -10
    io = open("result_iTEBD.txt", "a+")
    while convergence_flag <= 0
        _, Tpsi = left_canonical(act(T, psi))
        _, psi = iTEBD_truncate(Tpsi, chi)
        #psi = variational_truncate(Tpsi, chi)

        F_value = free_energy(T, psi)
        F_value2 = free_energy(T2, act(T_adapter, psi))
        costfunc_value = nonherm_cost_func(T, toarray(psi))
        costfunc_value2 = nonherm_cost_func(T2, toarray(act(T_adapter, psi)))
        fidelity_value = ln_fidelity(psi, Tpsi)

        if isapprox(costfunc_value, -fidelity_value; rtol=0.02, atol=1e-13)
            convergence_flag += 1
        end

        msg = "$chi $F_value $F_value2 $costfunc_value $fidelity_value $costfunc_value2"

        println(msg)
        println(io, msg)
    end
    close(io)
end

expct(T, psi) |> norm
expct(T2, act(T_adapter, psi)) |> norm

psi_arr = toarray(psi)

svd(psi_arr[:, 1, :] * psi_arr[:, 2, :])

T_arr = toarray(T)

for ix in 1:4
    _, s, _ = svd(T_arr[:, ix, 1, :]) 
    println("$(ix), 1 ", s)
end

for ix in 1:4
    _, s, _ = svd(T_arr[:, 1, ix, :]) 
    println("1, $(ix) ", s)
end 

t = TensorMap(zeros, ComplexF64, ℂ^4*ℂ^4, ℂ^4)
t[1, 2, 3] = 1
t[3, 1, 2] = 1
t[2, 3, 1] = 1
t[3, 2, 4] = 1
t[2, 4, 3] = 1
t[4, 3, 2] = 1

t_arr = toarray(t)
permutedims(t_arr, (1, 3, 2)) ≈ t_arr

psi
Ta, Tb = mpo_triangular_AF_ising_adapter()
psi1 = act(Tb, act(Ta, psi))

free_energy(T, psi)
free_energy(T, psi1)

psi1 = act(Ta, psi)

n, Γ, Λ = lambda_gamma(psi1)
Λ
s = diag(Λ.data).^2

psi1 = act(Ta, psi)
_, psi1 = iTEBD_truncate(act(Ta, psi), 32)

nonherm_cost_func_value = nonherm_cost_func(T, toarray(psi1))

sum(s .> 1e-16)

s = entanglement_spectrum(psi1)