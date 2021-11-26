using TensorKit
using Zygote
using LinearAlgebra
using Optim
using JLD2
using Random

using Revise
using statmech_tm_solver

# MPO for the triangular AF Ising
T = mpo_triangular_AF_ising()
T2 = mpo_triangular_AF_ising_alternative()
_, T_adapter = mpo_triangular_AF_ising_adapter()

function f(arr::Array{ComplexF64, 3})
    return nonherm_cost_func(T, arr_to_TensorMap(arr))
end

function g!(grad::Array{ComplexF64, 3}, psidata::Array{ComplexF64, 3})
    grad .= gradient(f, psidata)[1]
end

psidata = rand(MersenneTwister(1), ComplexF64, (2, 2, 2))

#io = open("result_variational.txt", "w")
#close(io)
for chi in [4, 8, 16, 32] 
    io = open("result_variational.txt", "a+")

    res_f = optimize(f, g!, psidata, LBFGS(), Optim.Options(show_trace=true, iterations=200))

    psidata_final = Optim.minimizer(res_f)
    psi_final = arr_to_TensorMap(psidata_final)

    cost_func_final = nonherm_cost_func(T, psi_final)
    F_final = free_energy(T, psi_final)
    F_final_2 = free_energy(T2, act(T_adapter, psi_final))
    cost_func_final2 = nonherm_cost_func(T2, act(T_adapter, psi_final))
    println(io, chi, ' ', F_final, ' ', F_final_2, ' ', cost_func_final, ' ', cost_func_final2 )
    quicksave("ckpt_variational_chi$(chi)", psi_final)    

    psi = act(T, psi_final)
    psidata = toarray(psi)

    close(io)
end

