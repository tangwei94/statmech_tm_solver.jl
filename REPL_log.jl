# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using TensorKit
using TensorOperations
using TensorKitAD
using KrylovKit
using Zygote
using KrylovKit
using LinearAlgebra
using Plots
using Optim
using ChainRules
using ChainRulesCore

using Revise
using statmech_tm_solver

ψ = cmps(rand, 4, 1)
V = TensorMap(rand, ComplexF64, ℂ^4, ℂ^4)

lop = lieb_liniger_h_tangent_map(ψ, 0, 0, 1.0, 100, 1)
lop(V)

A = TensorMap(rand, ComplexF64, ℂ^4, ℂ^4)
B = TensorMap(rand, ComplexF64, ℂ^4, ℂ^4)
Wvec = rand(4) + im .* rand(4)

L = 1
 

#########################################################
f1 = p -> (sqrt((p^2 - μ)^2 - 4 * ν^2) - (p^2 - μ)) / 2

Eexact = map(f1, (-100000:100000) .* 2*pi/L ) .* (1 / L) |> sum

@show (E - Eexact) / Eexact
