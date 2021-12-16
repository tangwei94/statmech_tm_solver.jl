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
T_adj = reshape(T.data, (2, 2, 2, 2))
T_adj = permutedims(conj.(T_adj), (1, 3, 2, 4))
T_adj = TensorMap(T_adj, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

function f(arr::Array{ComplexF64, 3})
    phi = arr_to_TensorMap(arr)
    #phi = act(Tn, phi)
    return nonherm_cost_func(T, phi)
end

function fT(arr::Array{ComplexF64, 3})
    phi = arr_to_TensorMap(arr)
    return nonherm_cost_func(T_adj, phi)
end

function g!(grad::Array{ComplexF64, 3}, psidata::Array{ComplexF64, 3})
    grad .= gradient(f, psidata)[1]
end

function gT!(grad::Array{ComplexF64, 3}, psidata::Array{ComplexF64, 3})
    grad .= gradient(fT, psidata)[1]
end

psidata = rand(MersenneTwister(1), ComplexF64, (2, 2, 2))

#io = open("result_variational.txt", "w")
#close(io)
for chi in [2, 4, 8, 16, 32, 64, 128, 256]
    global psidata 
    io = open("result_variational.txt", "a+")

    println(size(psidata))

    res_f = optimize(f, g!, psidata, LBFGS(), Optim.Options(show_trace=true, iterations=200))
    res_fT = optimize(fT, gT!, psidata, LBFGS(), Optim.Options(show_trace=true, iterations=200))

    psidata_final = Optim.minimizer(res_f)
    psi = arr_to_TensorMap(psidata_final)

    psiLdata_final = Optim.minimizer(res_fT)
    psiL = arr_to_TensorMap(psiLdata_final)

    cost_func_final = nonherm_cost_func(T, psi)
    F_final_1 = free_energy(T, psi)
    F_final = ovlp(psiL, act(T, psi)) / ovlp(psiL, psi) |> norm |> log
    println(io, chi, ' ', F_final, ' ', F_final_1, ' ', cost_func_final )
    quicksave("ckpt_variational_chi$(chi)", psi)    
    quicksave("ckpt_variational_left_chi$(chi)", psiL)    

    psidata = toarray(act(T, psi))

    close(io)
end
