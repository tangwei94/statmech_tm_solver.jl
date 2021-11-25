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

"""
    mpo_triangular_AF_ising_adapter()

    generate an MPO that converts the boundary MPS from the basis of `mpo_triangular_AF_ising_alternative()` to the basis of `mpo_triangular_AF_ising()`
"""
function mpo_triangular_AF_ising_adapter()
    p = TensorMap(zeros, ComplexF64, ℂ^4*ℂ^4, ℂ^4)
    m = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^4*ℂ^4)

    p[1, 1, 1] = p[2, 2, 2] = p[3, 3, 3] = p[4, 4, 4] = 1
    m[1, 1, 1] = m[1, 1, 2] = m[1, 3, 1] = m[1, 3, 2] = 1
    m[2, 2, 3] = m[2, 2, 4] = m[2, 4, 3] = m[2, 4, 4] = 1
    @tensor T[-1, -2; -3, -4] := p[-1, 1, -3] * m[-2, 1, -4]
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