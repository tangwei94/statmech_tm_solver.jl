# algorithm 4 in section 5.4 of https://www.annualreviews.org/doi/10.1146/annurev-conmatphys-031016-025507 applied to cMPO

struct qbimps
    A::TensorMap{ComplexSpace, 2, 1}
    B::cmps
end

function qbimps(f, chi::Integer, dp::Integer, dv::Integer)
    A = TensorMap(f, ComplexF64, ℂ^chi*ℂ^dp, ℂ^chi)
    B = cmps(f, chi, dv)
    return qbimps(A, B)
end

function A_canonical(T::cmpo, psi::qbimps)
    _, AR = right_canonical_QR(psi.A, 1e-12)

    chi = get_chi(psi.B)
    d = get_d(psi.B)

    BR_R = TensorMap(rand, ComplexF64, ℂ^chi*ℂ^d, ℂ^chi)
    BR_Q = TensorMap(rand, ComplexF64, ℂ^chi, ℂ^chi)

    function lopR(vR::TensorMap{ComplexSpace, 2, 1})
        @tensor TvR[-1, -2; -3] := vR[1, 3, 4] * AR[-1, 2, 1] * T.P[-2, 5, 2, 3] * AR'[4, -3, 5] +
                                   (-1.0) * vR[-1, -2, -3]
        return TvR
    end
    @tensor bR[-1, -2; -3] := AR[-1, 2, 1] * T.R[3, -2, 2] * AR'[1, -3, 3]
    BR_R, _ = linsolve(lopR, -bR, BR_R)

    lopQ = transf_mat(AR, AR)
    lopQ_T = transf_mat_T(AR, AR)
    @tensor bQ[-1; -2] := BR_R[1, 3, 4] * AR[-1, 2, 1] * AR'[4, -2, 5] * T.L'[5, 2, 3] +
                          AR[-1, 1, 2] * T.Q[3, 1] * AR'[2, -2, 3]
    Id_Q = id(ℂ^chi)
    _, vQL = eigsolve(lopQ_T, Id_Q, 1)
    vQL = vQL[1] / tr(vQL[1] * Id_Q')
    BR_Q, _ = linsolve(v -> lopQ(v)-tr(vQL'*v)*Id_Q-v, -bQ, BR_Q)

    # improve the linsolve result by a few more power iterations
    ix, δ = 0, 999
    while δ > 1e-12 && ix < 100 
        BR_Q1 = lopQ(BR_Q) - tr(vQL'*BR_Q)*Id_Q + bQ
        δ = (BR_Q1 - BR_Q).data / chi*2 |> norm 
        BR_Q = BR_Q1
        ix += 1
    end

    Λ = - tr(vQL' * BR_Q)
    (δ > 1e-12) && @warn "fixed point equation not fully converged for psi.B. δ=$δ"

    BR = cmps(BR_Q, BR_R)

    println("A canonical. Λ = $Λ")
    return qbimps(AR, BR)
end

function B_canonical(T::cmpo, psi::qbimps)
    _, BL = left_canonical(psi.B)

    #Xinv = inv(X)
    #@tensor AL[-1, -2; -3] := psi.A[1, -2, 2] * X[-1, 1] * Xinv[2, -3]
    chi, d = get_chi(psi.A), get_d(psi.A)
    AL = TensorMap(rand, ComplexF64, ℂ^chi*ℂ^d, ℂ^chi)

    ψL = BL
    @tensor result[-1; -2] := ψL.Q'[-1, -2] + ψL.Q[-1, -2] + ψL.R[1, 2, -2] * ψL.R'[-1, 1, 2]
    !isapprox(result, TensorMap(zeros, ComplexF64, ℂ^chi, ℂ^chi), atol=sqrt(eps())) && println("warning!!! ", result)

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

    w, AL = eigsolve(lop, AL, 1, :LR)
    AL = AL[1]
    println("B canonical. Λ = ", -w[1])

    return qbimps(AL, BL)
end

