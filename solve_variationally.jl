using Revise
using statmech_tm_solver
using TensorKit
using Zygote
using LinearAlgebra
using Optim

function cost_func_grad!(grad::Array{ComplexF64, 3}, psidata::Array{ComplexF64, 3})
    grad .= gradient(arr->nonherm_cost_func(T, arr), psidata)[1]
end

# MPO for the triangular AF Ising
t = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2)
p = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2*ℂ^2)
t[1, 1, 2] = 1
t[1, 2, 1] = 1
t[2, 1, 1] = 1
t[2, 2, 1] = 1
t[2, 1, 2] = 1
t[1, 2, 2] = 1
p[1, 1, 1] = 1
p[2, 2, 2] = 1

T = t*p
psidata = rand(ComplexF64, (2, 2, 2))
grad_tmp = rand(ComplexF64, (2, 2, 2))

io = open("result.txt", "w+")
for chi in [2, 4, 8] 
    res_f = optimize(arr->nonherm_cost_func(T, arr), cost_func_grad!, psidata, LBFGS(), Optim.Options(show_trace=true, iterations=200))
    psidata_final = Optim.minimizer(res_f)
    psi_final = arr_to_TensorMap(psidata_final)
    F_final = ovlp(psi_final, act(T, psi_final)) / ovlp(psi_final, psi_final)
    println(io, size(psidata), ' ', log(F_final |> real))
    psi = act(T, psi_final)
    psidata = reshape(psi.data, (2*chi, 2, 2*chi))
end
close(io)