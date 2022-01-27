# collection of quantum models to be solved by continuous tensor network methods

# cmpo representation for quantum models

"""
    cmpo_ising(Γ::Real)
"""
function cmpo_ising(Γ::Real)
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

"""
    cmpo_xy()
"""
function cmpo_xy()
    Q = zeros(ComplexF64, (2, 2))
    L = zeros(ComplexF64, (2, 2, 2))
    R = zeros(ComplexF64, (2, 2, 2))
    P = zeros(ComplexF64, (2, 2, 2, 2))

    σp = zeros(ComplexF64, (2, 2))
    σm = zeros(ComplexF64, (2, 2))
    σp[1, 2] = 1
    σm[2, 1] = 1


    L[:, 1, :] = σp
    L[:, 2, :] = σm

    R[:, 1, :] = σp 
    R[:, 2, :] = σm

    Q = convert_to_tensormap(Q, 1)
    L = convert_to_tensormap(L, 2)
    R = convert_to_tensormap(R, 2)
    P = convert_to_tensormap(P, 2)

    return cmpo(Q, L, R, P)
end

"""
    cmpo_xz()
"""
function cmpo_xz()
    Q = zeros(ComplexF64, (2, 2))
    L = zeros(ComplexF64, (2, 2, 2))
    R = zeros(ComplexF64, (2, 2, 2))
    P = zeros(ComplexF64, (2, 2, 2, 2))

    σx = zeros(ComplexF64, (2, 2))
    σz = zeros(ComplexF64, (2, 2))
    σx[1, 2] = 0.5
    σx[2, 1] = 0.5
    σz[1, 1] = 0.5
    σz[2, 2] = -0.5

    L[:, 1, :] = σx
    L[:, 2, :] = σz

    R[:, 1, :] = σx 
    R[:, 2, :] = σz

    Q = convert_to_tensormap(Q, 1)
    L = convert_to_tensormap(L, 2)
    R = convert_to_tensormap(R, 2)
    P = convert_to_tensormap(P, 2)

    return cmpo(Q, L, R, P)
end

"""
    energy_quantum_ising(psi::TensorMap{ComplexSpace, 2, 1}, Γ::Number)
"""
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

"""
    cmpo_xxz(Δ::Number)
"""
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

"""
    energy_quantum_xxz(psi::TensorMap{ComplexSpace, 2, 1}, Δ::Real)
"""
function energy_quantum_xxz(psi::TensorMap{ComplexSpace, 2, 1}, Δ::Real)
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

"""
    cmpo_ising_realtime(Γ::Real)
"""
function cmpo_ising_realtime(Γ::Real)
    Q = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    L = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^1, ℂ^2)
    R = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^1, ℂ^2)
    P = TensorMap(zeros, ComplexF64, ℂ^1*ℂ^2, ℂ^2*ℂ^1)

    Q[1, 2] = Γ * -1im
    Q[2, 1] = Γ * -1im
    L[1, 1, 1] = 1.0
    L[2, 1, 2] = -1.0
    R[1, 1, 1] = 1.0 * (-1im)
    R[2, 1, 2] = -1.0 * (-1im)

    return cmpo(Q, L, R, P)
end

"""
    energy_lieb_linger(ψ::cmps, c::Real, L::Real)

    Calculate the energy density of Lieb Linger model for cMPS `ψ` with system size `L`.
    `c` is the parameter in the Hamiltonian.
    The Hamiltonian of Lieb Liniger model
    ```
        H = ∫dx [(dψ† / dx)(dψ / dx) + cψ†ψ†ψψ - μψ†ψ]
    ```
    Note L can be set to `Inf`, which means the thermodynamic limit.
"""
function energy_lieb_liniger(ψ::cmps, c::Real, L::Real, μ::Real)
    tensorK = kinetic(ψ)
    tensorD = particle_density(ψ)
    tensorP = point_interaction(ψ)
    tensorE = tensorK + c * tensorP - μ * tensorD

    # calculate environment
    lop = K_mat(ψ, ψ)
    env = finite_env(lop, L)
    env = permute(env, (2, 3), (4, 1))
        
    return real(tr(env * tensorE))
end