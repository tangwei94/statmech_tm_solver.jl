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
    for ix in 1:100
        _, Tpsi = left_canonical(act(T, psi))
        _, psi = iTEBD_truncate(Tpsi, chi)
        #psi = variational_truncate(Tpsi, chi)
        println(chi, ' ', free_energy(T, psi), ' ', nonherm_cost_func(T, toarray(psi)),' ', ln_fidelity(psi, Tpsi))
        println(io, chi, ' ', free_energy(T, psi), ' ', nonherm_cost_func(T, toarray(psi)), ' ', ln_fidelity(psi, Tpsi))
    end
    
    _, psi = left_canonical(act(T, psi)) 

end

close(io)
