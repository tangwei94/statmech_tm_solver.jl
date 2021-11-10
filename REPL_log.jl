# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit

struct bimps
    A::TensorMap{ComplexSpace, 2, 1}
    B::TensorMap{ComplexSpace, 2, 1}
end

function bimps(f, chi::Integer, d::Integer)
    A = TensorMap(f, ComplexF64, ℂ^chi*ℂ^d, ℂ^chi)
    B = TensorMap(f, ComplexF64, ℂ^chi*ℂ^d, ℂ^chi)
    return bimps(A, B)
end

T = mpo_square_ising(0.5)
psi = bimps(rand, 4, 2)

A = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4)
B = TensorMap(rand, ComplexF64, ℂ^4*ℂ^2, ℂ^4)

bimps(A, B)

function A_canonical(T::TensorMap{ComplexSpace, 2, 2}, psi::bimps)

    # convert A to left canonical form
    X, AL = left_canonical_QR(psi.A, 1e-12)
    Xinv = inv(X)
    @tensor BL[-1, -2; -3] := psi.B[1, -2, 2] * X[-1, 1] * Xinv[2, -3] 

    # construct the linear operator for the maping of B
    function lop(v::TensorMap{ComplexSpace, 2, 1}) 
        @tensor Tv[-1, -2; -3] := v[1, 3, 4] * AL'[-1, 1, 2] * AL[4, 5, -3] * T[-2, 2, 5, 3]
        return Tv 
    end 

    # solve BL from the fixed point equation
    _, BL = eigsolve(lop, BL, 1)
    BL = BL[1]
    return bimps(AL, BL)
end

A_canonical(T, psi)

function B_canonical(T::TensorMap{ComplexSpace, 2, 2}, psi::bimps)

    # convert B to right canonical form
    Y, BR = right_canonical_QR(psi.B, 1e-12)
    Yinv = inv(Y')'
    @tensor AR[-1, -2; -3] := Yinv'[-1, 1] * psi.A[1, -2, 2] * Y'[2, -3]

    # construct the linear operator for the maping of A
    function lop(v::TensorMap{ComplexSpace, 2, 1}) 
        @tensor Tv[-1, -2; -3] := AR[1, 2, 4] * BR[-1, 3, 1] * T[5, -2, 2, 3] * BR'[4, -3, 5]
        return Tv 
    end 

    # solve AR from the fixed point equation
    _, AR = eigsolve(lop, AR, 1)
    AR = AR[1]
    return bimps(AR, BR)
end

psi = B_canonical(T, psi)

psi = bimps(rand, 8, 2)

psi = A_canonical(T, psi)
psi = B_canonical(T, psi)
println(free_energy(T, psi.A))
psi = A_canonical(T, psi)
#psi = B_canonical(T, psi)
# convert B to right canonical form
Y, BR = right_canonical_QR(psi.B, 1e-12)
Yinv = inv(Y')'
@tensor AR[-1, -2; -3] := Yinv'[-1, 1] * psi.A[1, -2, 2] * Y'[2, -3]

# construct the linear operator for the maping of A
function lop(v::TensorMap{ComplexSpace, 2, 1}) 
    @tensor Tv[-1, -2; -3] := AR[1, 2, 4] * BR[-1, 3, 1] * T[5, -2, 2, 3] * BR'[4, -3, 5]
    return Tv 
end 

# solve AR from the fixed point equation
_, AR = eigsolve(lop, AR, 1)
AR = AR[1]