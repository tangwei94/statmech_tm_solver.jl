# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using TensorKit
using TensorOperations
using TensorKitAD
using Zygote
using KrylovKit
using LinearAlgebra
using Plots

using Revise
using statmech_tm_solver

A = quickload("data/triangular_AF_ising/checkpoints/ckpt_vumps_AL_chi32")
B = quickload("data/triangular_AF_ising/checkpoints/ckpt_vumps_BL_chi32")

@tensor M[-1, -2; -3, -4] := A[-2, 1, -4] * B'[-3, -1, 1]
W, V = eig(M)
Vinv = inv(V)
space(V)

V = permute(V, (2,), (1, 3))
Vinv = permute(Vinv, (1, 2), (3, ))

@show diag(W.data)

O = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
O[1, 1] = 1
O[2, 2] = -1
O

@tensor M1[-1, -2; -3, -4] := A[-2, 1, -4] * O[2, 1] * B'[-3, -1, 2]
M1 = permute(M1, (2, 3), (1, 4))

ER = convert_to_tensormap(convert_to_array(V)[:, :, end], 1)
EL = convert_to_tensormap(convert_to_array(Vinv)[end, :, :], 1)

@tensor ML[-1] := EL[3, 4] * M1[4, 1, 3, 2] * V[2, 1, -1]
ML = permute(ML, (), (1,))
@tensor MR[-1] := ER[4, 3] * M1[2, 3, 1, 4] * Vinv[-1, 1, 2]

@tensor EL[1, 2] * M1[2, 4, 1, 3] * ER[3, 4]

corrs = zeros(100)
for n in 1:100
    corr_n = (ML * W^n * MR / tr(W^n))[1] |> real
    corrs[n] = corr_n
    @show n, corr_n
end

gr()
plot(4:100, corrs[4:end])