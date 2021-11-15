# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using Revise
using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit

psi = qbimps(rand, 12, 2, 1)
T = cmpo_ising(1.0)
TA = mpo_quantum_ising(1.0)
psi = B_canonical(T, psi)
psi = A_canonical(T, psi)
println(exp(free_energy(TA, psi.A)))

for ix in 1:100
    psi = B_canonical(T, psi)
    psi = A_canonical(T, psi)
    

    println(exp(free_energy(TA, psi.A)))
end

TA = mpo_quantum_ising(0.0)
psi = qbimps(rand, 2, 2, 1)
ovlp(psi.A, act(TA, psi.A)) / ovlp(psi.A, psi.A)

dataA = rand(ComplexF64, 8, 2, 8)
function f(data)
    psiA = arr_to_TensorMap(data)
    return -ovlp(psiA, act(TA, psiA)) / ovlp(psiA, psiA) |> real
end
function g!(grad, data)
    grad .= gradient(f, data)[1]
end

f(dataA)

using Optim, Zygote
optimize(f, g!, dataA, LBFGS(), Optim.Options(show_trace=true, iterations=100))