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
T_adapter1, T_adapter = mpo_triangular_AF_ising_adapter()

function f(arr::Array{ComplexF64, 3})
    phi = arr_to_TensorMap(arr)
    #phi = act(Tn, phi)
    return nonherm_cost_func(T, phi)
end

function g!(grad::Array{ComplexF64, 3}, psidata::Array{ComplexF64, 3})
    grad .= gradient(f, psidata)[1]
end

psidata = rand(MersenneTwister(1), ComplexF64, (2, 2, 2))

io = open("result_variational.txt", "w")
close(io)
for chi in [2, 4, 8, 16]
    global psidata 
    io = open("result_variational.txt", "a+")

    println(size(psidata))

    res_f = optimize(f, g!, psidata, LBFGS(), Optim.Options(show_trace=true, iterations=200))

    psidata_final = Optim.minimizer(res_f)
    psi = arr_to_TensorMap(psidata_final)

    cost_func_final = nonherm_cost_func(T, psi)
    F_final = free_energy(T, psi)
    F_final_2 = free_energy(T2, act(T_adapter, psi))
    cost_func_final2 = nonherm_cost_func(T2, act(T_adapter, psi))
    println(io, chi, ' ', F_final, ' ', cost_func_final, ' ', F_final_2, ' ', cost_func_final2 )
    quicksave("ckpt_variational_chi$(chi)", psi)    

    psidata = toarray(act(T, psi))

    close(io)
end
## todo: z2 symmetry?
#@tensor Tlinked[-1, -3, -5, -7, -9, -11, -13, -15, -17, -19; -2, -4, -6, -8, -10, -12, -14, -16, -18, -20] := 
#    T[1, -1, -2, 2] * 
#    T[2, -3, -4, 3] *
#    T[3, -5, -6, 4] *
#    T[4, -7, -8, 5] *
#    T[5, -9, -10, 6] *
#    T[6, -11, -12, 7] *
#    T[7, -13, -14, 8] *
#    T[8, -15, -16, 9] * 
#    T[9, -17, -18, 10] * 
#    T[10, -19, -20, 1]; 
#
#D, V = eig(Tlinked);
#(D.data |> diag)
#for ix in 1:10
#    println(ix, ' ', log(abs(D[ix, ix])) / 10, ' ', reshape(toarray(V), (1024, 1024))[1, ix])
#end
#for ix in 1000:1024
#    println(ix, ' ', log(abs(D[ix, ix])) / 10, ' ', reshape(toarray(V), (1024, 1024))[1, ix])
#end