# calculate the spin correlation functions 
# using the fixed point MPS obtained from the non-frustrated MPO

# run the code in the root directory

using TensorKit
using TensorOperations
using TensorKitAD
using Zygote
using KrylovKit
using LinearAlgebra
using Plots

using Revise
using statmech_tm_solver

# load prev data
A = quickload("data/triangular_AF_ising/checkpoints/ckpt_iTEBD_chi32_alternative")

# mps transfer matrix
@tensor M[-1, -2; -3, -4] := A[-2, 1, -4] * A'[-3, -1, 1]
W, V = eig(M)
Vinv = inv(V)
space(V)

V = permute(V, (2,), (1, 3))
Vinv = permute(Vinv, (1, 2), (3, ))

@show diag(W.data)

# observable 
O = TensorMap(zeros, ComplexF64, ℂ^4, ℂ^4)
O[1, 1] = 1
O[2, 2] = 1
O[3, 3] = -1
O[4, 4] = -1
@show O

@tensor M1[-1, -2; -3, -4] := A[-2, 1, -4] * O[2, 1] * A'[-3, -1, 2]
M1 = permute(M1, (2, 3), (1, 4))

# environment
ER = convert_to_tensormap(convert_to_array(V)[:, :, end], 1)
EL = convert_to_tensormap(convert_to_array(Vinv)[end, :, :], 1)

@tensor ML[-1] := EL[3, 4] * M1[4, 1, 3, 2] * V[2, 1, -1]
ML = permute(ML, (), (1,))
@tensor MR[-1] := ER[4, 3] * M1[2, 3, 1, 4] * Vinv[-1, 1, 2]

# local observable <Z>
@tensor EL[1, 2] * M1[2, 4, 1, 3] * ER[3, 4]

# spin correlation, quasi-long-range order observed
corrs = zeros(300)
for n in 1:300
    corr_n = (ML * W^n * MR / tr(W^n))[1] |> real
    corrs[n] = corr_n
    @show n, corr_n
end

# log log plot of the spin correlation
gr()
plot(3:300, corrs[2:end])

plot(xs, ys)