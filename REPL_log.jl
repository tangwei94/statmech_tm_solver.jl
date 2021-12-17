# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using TensorKit
using TensorOperations
using KrylovKit
using LinearAlgebra
using Zygote
using Revise
using statmech_tm_solver

psi = quickload("tmp/ckpt_variational_chi32")
T = mpo_triangular_AF_ising()

_, psiR = right_canonical_QR(psi)

@tensor tm[-1, -2, -3; -4, -5, -6] := psiR[-1, 1, -4] * T[-2, 2, 1, -5] * psiR'[-6, -3, 2]

w, _ = eig(tm)
w = diag(w.data)

w_reals, w_imags = real.(w), imag.(w)

io = open("tmp.txt", "w")
for (w_real, w_imag) in zip(w_reals, w_imags)
    println(io, "$(w_real)    $(w_imag)")
end
close(io)

#######################################################################

psi = quickload("tmp/ckpt_variational_chi32")
@tensor tm[-1, -2; -3, -4] := psi[-1, 1, -3] * psi'[-4, -2, 1]
w, _ = eig(tm);
w = diag(w.data)

w_reals, w_imags = real.(w), imag.(w)

io = open("tmp.txt", "w")
for (w_real, w_imag) in zip(w_reals, w_imags)
    println(io, "$(w_real)    $(w_imag)")
end
close(io)

w1, _ = eig(tm * exp(-2*pi/3*im))
w1 = diag(w1.data)
w


#######################################################################

T = mpo_triangular_AF_ising()

A_fixed = quickload("tmp/ckpt_variational_chi32")
psi = bimps(A_fixed, TensorMap(rand, ComplexF64, ℂ^32*ℂ^2, ℂ^32))

free_energy(T, psi.A)

for ix in 1:20
    λA, psi = A_canonical(T, psi); @show angle(λA)/pi*3
    λB, psi = B_canonical(T, psi); @show angle(λB)/pi*3
    @show free_energy(T, psi.A)
end
