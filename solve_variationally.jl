using Revise
using statmech_tm_solver
using TensorKit
using Zygote
using LinearAlgebra
using Optim

# MPO for the triangular AF Ising
T = mpo_triangular_AF_ising()
T2 = mpo_triangular_AF_ising_alternative()
_, T_adapter = mpo_triangular_AF_ising_adapter()

function cost_func_grad!(grad::Array{ComplexF64, 3}, psidata::Array{ComplexF64, 3})
    grad .= gradient(arr->nonherm_cost_func(T, arr), psidata)[1]
end

psidata = rand(ComplexF64, (2, 2, 2))

io = open("result.txt", "w")
close(io)
for chi in [2, 4, 8, 16, 32] 
    io = open("result.txt", "a+")

    res_f = optimize(arr->nonherm_cost_func(T, arr), cost_func_grad!, psidata, LBFGS(), Optim.Options(show_trace=true, iterations=200))
    psidata_final = Optim.minimizer(res_f)
    psi_final = arr_to_TensorMap(psidata_final)
    cost_func_final = nonherm_cost_func(T, psidata_final)
    F_final = free_energy(T, psi_final)
    F_final_2 = free_energy(T2, act(T_adapter, psi_final))
    println(io, chi, ' ', F_final, ' ', F_final_2, ' ', cost_func_final )
    psi = act(T, psi_final)
    psidata = reshape(psi.data, (2*chi, 2, 2*chi))

    close(io)
end

