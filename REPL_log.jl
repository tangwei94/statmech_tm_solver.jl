# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using TensorKit
using TensorOperations
using TensorKitAD
using KrylovKit
using LinearAlgebra
using Zygote
using Revise
using statmech_tm_solver

psi = cmps(rand, 4, 3)
phi = cmps(rand, 4, 3)
op = cmpo_xxz(0.5)

gradient(x -> real(log_ovlp(act(op, x), phi, 4)), psi)[1]

a = rand(ComplexF64, 4, 4, 4)
a_cmps = convert_to_cmps(a)
function f(a::Array{ComplexF64})
    a_cmps = convert_to_cmps(a)
    return real(log_ovlp(a_cmps, a_cmps, 4.))
end
f'(a)


size_arr = size(a[:, 1, :])
d_arr = length(size_arr)
d_dom = 1
dom = ifelse(d_dom > 0, prod(map(x->ℂ^x, size_arr[1:d_dom])), ProductSpace{ComplexSpace, 0})
prod(map(x->ℂ^x, size_arr[1:d_dom]))
size_arr[1:d_dom]
map(x->ℂ^x, (4,))
prod((ℂ^4,ℂ^4))

##################

a = rand(ComplexF64, (2,3,4,5,6))
size_a = size(a)
d_dom, d_codom = 3, 2
dom = ProductSpace{ComplexSpace, 0}
codom = ProductSpace{ComplexSpace, 0}
map(x->ℂ^x, size_a[1:3]) |> prod

##################

Id_psi = id(ℂ^8)

function log_ovlp3(Q::TensorMap{ComplexSpace})
    @tensor t_trans[-1, -2; -3, -4] := Q[-1, -3] * Id_psi[-2, -4] 
    return norm(t_trans)
end

ψ = cmps(rand, 4, 2)

log_ovlp3(ψ.Q)
gradient(x -> log_ovlp3(x), TensorMap(rand, ComplexF64, ℂ^4, ℂ^4))
