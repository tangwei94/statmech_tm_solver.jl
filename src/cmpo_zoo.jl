function cmpo_ising(Γ::Number)
    Q = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    L = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^1, ℂ^2)
    R = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^1, ℂ^2)
    P = TensorMap(zeros, ComplexF64, ℂ^1*ℂ^2, ℂ^2*ℂ^1)

    Q[1, 2] = Γ
    Q[2, 1] = Γ
    L[1, 1, 1] = 1.0
    L[2, 1, 2] = -1.0
    R[1, 1, 1] = 1.0
    R[2, 1, 2] = -1.0

    return cmpo(Q, L, R, P)
end

function energy_quantum_ising(psi::TensorMap{ComplexSpace, 2, 1}, Γ::Number)
    σx = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    σz = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)

    σx[1, 2] = 1
    σx[2, 1] = 1

    σz[1, 1] = 1
    σz[2, 2] = -1

    lop = transf_mat(psi, psi)
    lopT = transf_mat_T(psi, psi)
    chi = get_chi(psi)
    v0 = TensorMap(rand, ComplexF64, ℂ^chi, ℂ^chi)

    w, vr = eigsolve(lop, v0, 1)
    _, vl = eigsolve(lopT, v0, 1)
    w = w[1]
    vr = vr[1]
    vl = vl[1]
    vr = vr / tr(vl' * vr)

    @tensor term_1site[:] := vl'[1, 2] * psi'[5, 1, 3] * σx[3, 4] * psi[2, 4, 6] * vr[6, 5]
    @tensor term_2site[:] := vl'[1, 2] * psi'[5, 1, 3] * σz[3, 4] * psi[2, 4, 6] * psi'[9, 5, 7] * σz[7, 8] * psi[6, 8, 10] * vr[10, 9]

    term_1site = term_1site / w * (-Γ)
    term_2site = term_2site / w^2 * (-1)

    return (term_1site + term_2site)[1]
end

function cmpo_xxz(Δ::Number)
    Q = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    L = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^3, ℂ^2)
    R = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^3, ℂ^2)
    P = TensorMap(zeros, ComplexF64, ℂ^3*ℂ^2, ℂ^2*ℂ^3)

    L[1, 1, 2] = 1 / sqrt(2)
    L[2, 2, 1] = 1 / sqrt(2) 
    L[1, 3, 1] = -0.5*sqrt(abs(Δ))
    L[2, 3, 2] = 0.5*sqrt(abs(Δ))

    R[1, 1, 2] = 1 / sqrt(2)
    R[2, 2, 1] = 1 / sqrt(2)
    R[1, 3, 1] = 0.5*sqrt(abs(Δ))*sign(Δ)
    R[2, 3, 2] = -0.5*sqrt(abs(Δ))*sign(Δ)

    return cmpo(Q, L, R, P)
end

function energy_quantum_xxz(psi::TensorMap{ComplexSpace, 2, 1}, Δ::Number)
    sx = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    sy = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    sz = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)

    sx[1, 2] = 0.5
    sx[2, 1] = 0.5
    sy[1, 2] = 0.5im
    sy[2, 1] = -0.5im
    sz[1, 1] = 0.5
    sz[2, 2] = -0.5

    lop = transf_mat(psi, psi)
    lopT = transf_mat_T(psi, psi)
    chi = get_chi(psi)
    v0 = TensorMap(rand, ComplexF64, ℂ^chi, ℂ^chi)

    w, vr = eigsolve(lop, v0, 1)
    _, vl = eigsolve(lopT, v0, 1)
    w = w[1]
    vr = vr[1]
    vl = vl[1]
    vr = vr / tr(vl' * vr)

    @tensor term_xx[:] := vl'[1, 2] * psi'[5, 1, 3] * sx[3, 4] * psi[2, 4, 6] * psi'[9, 5, 7] * sx[7, 8] * psi[6, 8, 10] * vr[10, 9]
    @tensor term_yy[:] := vl'[1, 2] * psi'[5, 1, 3] * sy[3, 4] * psi[2, 4, 6] * psi'[9, 5, 7] * sy[7, 8] * psi[6, 8, 10] * vr[10, 9]
    @tensor term_zz[:] := vl'[1, 2] * psi'[5, 1, 3] * sz[3, 4] * psi[2, 4, 6] * psi'[9, 5, 7] * sz[7, 8] * psi[6, 8, 10] * vr[10, 9]

    result = term_xx / w^2 * (-1) + term_yy / w^2 * (-1) + term_zz / w^2 * Δ

    return result[1]
end

function cmpo_ising_realtime(Γ::Number)
    Q = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    L = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^1, ℂ^2)
    R = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^1, ℂ^2)
    P = TensorMap(zeros, ComplexF64, ℂ^1*ℂ^2, ℂ^2*ℂ^1)

    Q[1, 2] = Γ * 1im
    Q[2, 1] = Γ * 1im
    L[1, 1, 1] = 1.0
    L[2, 1, 2] = -1.0
    R[1, 1, 1] = 1.0 * 1im
    R[2, 1, 2] = -1.0 * 1im

    return cmpo(Q, L, R, P)
end