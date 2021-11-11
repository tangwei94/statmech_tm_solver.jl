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

    # convert A to right canonical form
    Y, AR = right_canonical_QR(psi.A, tol)
    Yinv = inv(Y')'
    @tensor BR[-1, -2; -3] := psi.B[1, -2, 2] * Yinv'[-1, 1] * Y'[2, -3]  

    # construct the linear operator for the maping of B
    function lop(v::TensorMap{ComplexSpace, 2, 1}) 
        @tensor Tv[-1, -2; -3] := v[1, 3, 4] * AR'[4, -3, 5] * AR[-1, 2, 1] * T[-2, 5, 2, 3]
        return Tv 
    end 

    # solve BR from the fixed point equation
    _, BR = eigsolve(lop, BR, 1)
    BR = BR[1]
    return bimps(AR, BR)
end

function B_canonical(T::TensorMap{ComplexSpace, 2, 2}, psi::bimps, tol::Float64=1e-15)

    # convert B to left canonical form
    X, BL = left_canonical_QR(psi.B, tol)
    Xinv = inv(X)
    @tensor AL[-1, -2; -3] := X[-1, 1] * psi.A[1, -2, 2] * Xinv[2, -3]

    # construct the linear operator for the maping of A
    function lop(v::TensorMap{ComplexSpace, 2, 1}) 
        @tensor Tv[-1, -2; -3] := v[4, 2, 1] * BL[1, 3, -3] * T[5, -2, 2, 3] * BL'[-1, 4, 5]
        return Tv 
    end 

    # solve AR from the fixed point equation
    _, AL = eigsolve(lop, AL, 1)
    AL = AL[1]
    return bimps(AL, BL)
end

