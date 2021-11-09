using TensorKit
using TensorOperations
using statmech_tm_solver
using KrylovKit

psi = TensorMap(rand, ComplexF64, ℂ^6*ℂ^2, ℂ^6)
phi = TensorMap(rand, ComplexF64, ℂ^6*ℂ^2, ℂ^6)

_, psi_L = left_canonical(psi)
ln_fidelity(psi_L, psi)

function right_canonical_QR(psi::TensorMap{ComplexSpace, 2, 1}, tol::Float64=1e-15)

    chi = get_chi(psi)
    L0 = id(ℂ^chi)

    L, Q = rightorth(permute(psi * L0, (1, ), (2, 3)))
    psi_R = permute(Q, (1, 2), (3, ))
    L = L / norm(L)
    δ = norm(L - L0)
    L0 = L

    while δ > tol
        lop = transf_mat(psi, psi_R)
        _, vr = eigsolve(lop, L0, 1; tol=max(tol, δ/10))
        L = vr[1]' 

        L, Q = rightorth(permute(psi * L, (1, ), (2, 3)))
        psi_R = permute(Q, (1, 2), (3, ))
        L = L / norm(L)

        δ = norm(L-L0)
        L0 = L
        println(δ)

    end

    return L0', psi_R
end

function left_canonical_QR(psi::TensorMap{ComplexSpace, 2, 1}, tol::Float64=1e-15)
    chi = get_chi(psi)
    R0 = id(ℂ^chi)

    psi_L, R = leftorth(psi)
    R = R / norm(R)
    δ = norm(R - R0)
    R0 = R

    while δ > tol
        lop_T = transf_mat_T(psi, psi_L)
        _, vl = eigsolve(lop_T, R0, 1; tol=max(tol, δ/10))
        R = vl[1]

        @tensor psi_tmp[-1, -2; -3] := R[-1, 1] * psi[1, -2, -3]
        psi_L, R = leftorth(psi_tmp)
        R = R / norm(R)

        δ = norm(R - R0)
        R0 = R
        println(δ)
    end
    return R0, psi_L
end
