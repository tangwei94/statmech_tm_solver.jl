# algorithm 4 in section 5.4 of https://www.annualreviews.org/doi/10.1146/annurev-conmatphys-031016-025507

struct bimps
    A::TensorMap{ComplexSpace, 2, 1}
    B::TensorMap{ComplexSpace, 2, 1}
end

function bimps(f, chi::Integer, d::Integer)
    A = TensorMap(f, ComplexF64, ℂ^chi*ℂ^d, ℂ^chi)
    B = TensorMap(f, ComplexF64, ℂ^chi*ℂ^d, ℂ^chi)
    return bimps(A, B)
end

function A_canonical(T::TensorMap{ComplexSpace, 2, 2}, psi::bimps, tol::Float64=1e-15)

    # convert A to left canonical form
    X, AL = left_canonical_QR(psi.A, tol)
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

function B_canonical(T::TensorMap{ComplexSpace, 2, 2}, psi::bimps, tol::Float64=1e-15)

    # convert B to right canonical form
    Y, BR = right_canonical_QR(psi.B, tol)
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

