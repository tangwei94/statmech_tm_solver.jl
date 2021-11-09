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