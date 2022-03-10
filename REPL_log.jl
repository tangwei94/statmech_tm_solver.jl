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
using BenchmarkTools
using Tullio
using LoopVectorization

using Revise
using statmech_tm_solver


lop = lieb_liniger_h_tangent_map(ψ, 0, 0, 1.0, 100, 1)
lop(V)

A = TensorMap(rand, ComplexF64, ℂ^64, ℂ^64)
B = TensorMap(rand, ComplexF64, ℂ^64, ℂ^64)
Wvec = 0.01*(rand(64) + im .* rand(64))

L = 1
M = zero(A)
@benchmark @tullio M.data[i, k] = A.data[i, j] * B.data[j, k] * statmech_tm_solver.theta3(L, Wvec[i], Wvec[j], Wvec[k]) 
C = zeros(ComplexF64, (64, 64, 64))
@benchmark @tullio C[i, j, k] = statmech_tm_solver.theta3(L, Wvec[i], Wvec[j], Wvec[k])
@tullio C[i, j, k] = statmech_tm_solver.theta3(L, Wvec[i], Wvec[j], Wvec[k])

@benchmark statmech_tm_solver.elem_mult_f2(A, B, (ix, im, iy)->statmech_tm_solver.theta3(L, Wvec[ix], Wvec[im], Wvec[iy]))
@benchmark @tullio M.data[i, k] = A.data[i, j] * B.data[j, k] * C[i, j, k]

L = 1
χ = 12
ψ = cmps(rand, χ, 1);
V = TensorMap(rand, ComplexF64, ℂ^χ, ℂ^χ);
lop = tangent_map(ψ, L);
@benchmark lop(V)

hop = lieb_liniger_h_tangent_map(ψ, 0, 0, 1.0, 100, 1);
@benchmark hop(V)





#########################################################
f1 = p -> (sqrt((p^2 - μ)^2 - 4 * ν^2) - (p^2 - μ)) / 2

Eexact = map(f1, (-100000:100000) .* 2*pi/L ) .* (1 / L) |> sum

@show (E - Eexact) / Eexact
