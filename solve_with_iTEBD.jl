using Revise
using statmech_tm_solver
using TensorKit
using Zygote
using LinearAlgebra
using Optim

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

function cost_func_grad!(grad::Array{ComplexF64, 3}, psidata::Array{ComplexF64, 3})
    grad .= gradient(arr->nonherm_cost_func(T, arr), psidata)[1]
end

psi = TensorMap(rand, ComplexF64, ℂ^2*ℂ^2, ℂ^2)
_, psi = left_canonical(psi)

io = open("result.txt", "w+")
for chi in [2; 4; 8; 16]
    for ix in 1:20
        _, Tpsi = left_canonical(act(T, psi))
        Tpsi = mps_add(Tpsi, psi) + 1e-2 * TensorMap(rand, ComplexF64, ℂ^(3*chi)*ℂ^2, ℂ^(3*chi)) 
        _, psi = iTEBD_truncate(Tpsi, chi)
        println(ix, ' ', ln_free_energy(T, psi), ' ', ln_fidelity(psi, Tpsi))
    end
    
    if chi >= 8
        psi_arr = toarray(psi)
        res_f = optimize(arr->nonherm_cost_func(T, arr), cost_func_grad!, psi_arr, LBFGS(), Optim.Options(show_trace=true, iterations=200))
        psi = Optim.minimizer(res_f) |> arr_to_TensorMap
    end

    F = ln_free_energy(T, psi)
    println(io, get_chi(psi), ' ', F)
    _, psi = left_canonical(act(T, psi)) 

end
close(io)

#for chi in [2, 4, 8, 16, 32, 64] 
#end
