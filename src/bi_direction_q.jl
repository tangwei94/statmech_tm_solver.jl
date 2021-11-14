# algorithm 4 in section 5.4 of https://www.annualreviews.org/doi/10.1146/annurev-conmatphys-031016-025507

struct qbimps
    A::TensorMap{ComplexSpace, 2, 1}
    B::cmps
end

function qbimps(f, chi::Integer, dp::Integer, dv::Integer)
    A = TensorMap(f, ComplexF64, ℂ^chi*ℂ^dp, ℂ^chi)
    B = cmps(f, chi, dv)
    return qbimps(A, B)
end

function A_canonical(T::cmpo, psi::qbimps, tol::Float64=1e-15)
    Y, AR = right_canonical_QR(psi.A)
    Yinv = inv(Y')'

    @tensor BR_R[-1, -2; -3] := psi.B.R[1, -2, 2] * Yinv'[-1, 1] * Y'[2, -3]
    BR_Q = Yinv' * psi.B.Q * Y'

    function lopR(vR::TensorMap{ComplexSpace, 2, 1})
        @tensor TvR[-1, -2; -3] := vR[1, 3, 4] * AR[-1, 2, 1] * T.P[-2, 5, 2, 3] * AR'[4, -3, 5] +
                                   AR[-1, 2, 1] * T.R[3, -2, 2] * AR'[1, -3, 3]
        return TvR
    end
    _, BR_R = eigsolve(lopR, BR_R, 1)
    BR_R = BR_R[1]

    function lopQ(vQ::TensorMap{ComplexSpace, 1, 1})
        @tensor TvQ[-1; -2] := vQ[1, 2] * AR[-1, 3, 1] * AR'[2, -2, 3] + 
                               AR[-1, 1, 2] * T.Q[3, 1] * AR'[2, -2, 3] +
                               BR_R[1, 3, 4] * AR[-1, 2, 1] * AR'[4, -2, 5] * T.L'[5, 2, 3]
        return TvQ
    end
    _, BR_Q = eigsolve(lopQ, BR_Q, 1)
    BR_Q = BR_Q[1]

    BR = cmps(BR_Q, BR_R)

    return qbimps(AR, BR)
end

function B_canonical(T::cmpo, psi::qbimps, tol::Float64=1e-15)
    X, BL = left_canonical(psi.B)
    Xinv = inv(X)
    @tensor AL[-1, -2; -3] := psi.A[1, -2, 2] * X[-1, 1] * Xinv[2, -3]

    function lop(v::TensorMap{ComplexSpace, 2, 1})
        @tensor Tv[-1, -2; -3] := 
            v[-1, -2, 1] * BL.Q[1, -3] +
            v[-1, 1, -3] * T.Q[-2, 1] + 
            v[1, -2, -3] * BL.Q'[-1, 1] + 
            v[-1, 1, 2] * T.L'[-2, 1, 3] * BL.R[2, 3, -3] +
            v[1, 2, -3] * T.R[-2, 3, 2] * BL.R'[-1, 1, 3] + 
            v[1, 2, 4] * BL.R[4, 5, -3] * T.P[3, -2, 2, 5] * BL.R'[-1, 1, 3]
        return Tv
    end

    _, AL = eigsolve(lop, AL, 1)
    AL = AL[1]

    return qbimps(AL, BL)
end

