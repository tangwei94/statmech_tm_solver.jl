function mpo_triangular_AF_ising()
    # exact: 0.3230659669
    # ref: Phys. Rev. Res. 3, 013041 (2021)
    t = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2)
    p = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2*ℂ^2)
    t[1, 1, 2] = 1
    t[1, 2, 1] = 1
    t[2, 1, 1] = 1
    t[2, 2, 1] = 1
    t[2, 1, 2] = 1
    t[1, 2, 2] = 1
    p[1, 1, 1] = 1
    p[2, 2, 2] = 1
    T = t*p
    return T
end

function mpo_triangular_AF_ising_alternative()
    # exact: 0.3230659669
    # ref: Phys. Rev. Res. 3, 013041 (2021)
    t = TensorMap(zeros, ComplexF64, ℂ^4*ℂ^4, ℂ^4)
    p = TensorMap(zeros, ComplexF64, ℂ^4, ℂ^4)
    t[1, 2, 3] = 1
    t[3, 1, 2] = 1
    t[2, 3, 1] = 1
    t[3, 2, 4] = 1
    t[2, 4, 3] = 1
    t[4, 3, 2] = 1
    p[1, 1] = 1
    p[2, 2] = 1
    p[3, 3] = 1
    p[4, 4] = 1
    @tensor T[-1, -2; -3, -4] := t'[3,1,2] * p[-2,3] * p[1,4] * p[2,-4] * t[-1,4,-3]
    return T
end

function mpo_square_ising(beta::Float64)
    # ising model on square lattice
    δ = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
    t = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    δ[1, 1, 1, 1] = 1
    δ[2, 2, 2, 2] = 1
    t[1, 1] = exp(-beta)
    t[2, 2] = exp(-beta)
    t[1, 2] = exp(beta)
    t[2, 1] = exp(beta)

    u, s, v = tsvd(t)
    u = u * sqrt(s)
    v = sqrt(s) * v

    @tensor T[-1, -2; -3, -4] := δ[1, 2, 3, 4] * v[-1, 1] * v[-2, 2] * u[3, -3] * u[4, -4]
    return T
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

    @tensor term1[:] := vl'[1, 2] * psi'[5, 1, 3] * σx[3, 4] * psi[2, 4, 6] * psi'[8, 5, 7] * psi[6, 7, 9] * vr[9, 8]
    @tensor term2[:] := vl'[1, 2] * psi'[4, 1, 3] * psi[2, 3, 5] * psi'[8, 4, 6] * σx[6, 7] * psi[5, 7, 9] * vr[9, 8]
    @tensor term3[:] := vl'[1, 2] * psi'[5, 1, 3] * σz[3, 4] * psi[2, 4, 6] * psi'[9, 5, 7] * σz[7, 8] * psi[6, 8, 10] * vr[10, 9]

    return (-Γ*term1/2 - Γ*term2/2 - term3)[1] / tr(vl' * vr) / w^2 

end
