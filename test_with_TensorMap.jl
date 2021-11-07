

function abs_diag(m::TensorMap{CartesianSpace, 1, 1})
    m_diag = Diagonal(m.data)
    chi = dim(domain(m))

    factor = sign(m_diag[1, 1])
    for ix in 1:chi
        m_diag[ix, ix] = abs(m_diag[ix, ix])
    end
    return TensorMap(Matrix(m_diag), ℝ^chi, ℝ^chi)
end

function pseudo_inv(m::TensorMap{CartesianSpace, 1, 1})
    m_diag = Diagonal(m.data)
    chi = dim(domain(m))
    for ix in 1:chi
        if abs(m_diag[ix, ix]) > 1e-8
            m_diag[ix, ix] = 1 / abs(m_diag[ix, ix])
        else
            m_diag[ix, ix] = 0    
        end
    end
    return TensorMap(Matrix(m_diag), ℝ^chi, ℝ^chi)
end

V = [1; 2; 3e-10]
a_diag = diagm(V)
a_diag = TensorMap(a_diag, ℝ^3, ℝ^3)
pseudo_inv(a_diag)

##### power method, use iTEBD
function idmrg(psi::TensorMap{CartesianSpace, 2, 1}, target_chi::Integer)
    lop = transf_mat(psi, psi)
    lopT = transf_mat_T(psi, psi)
    chi = dim(domain(psi))
    v0 = Tensor(rand, Float64, ℝ^chi*ℝ^chi)

    wr, vr = eigsolve(lop, v0, 1)
    wr = wr[1]
    vr = vr[1] / vr[1][1,1]

    _, vl = eigsolve(lopT, v0, 1)
    vl = vl[1] / vl[1][1,1]

    vr = permute(vr, (1,), (2,))
    vl = permute(vl, (1,), (2,))
    @assert vr ≈ vr'
    @assert vl ≈ vl'

    λr, Ur = eigh(vr)
    for ix in 1:chi
        @assert λr[ix, ix] * λr[1,1] > 0
    end
    λr = abs_diag(λr)
    Xr = Ur * sqrt(λr)
    Xr_inv = sqrt(pseudo_inv(λr)) * Ur'

    λl, Ul = eigh(vl)
    λl = abs_diag(λl)
    Xl = Ul * sqrt(λl)
    Xl_inv = sqrt(pseudo_inv(λl)) * Ul'

    λ1 = permute(Xl, (2,), (1,)) * Xr
    Γ1 = @ncon((Xr_inv, psi, Xl_inv'), 
        ([-1, 1], [1, -2, 2], [2, -3])
    )
    Γ1 = permute(Γ1, (1, 2), (3,))

    u, λ2, v = tsvd(λ1)
    λ2 /= sqrt(real(abs(wr))) # renormalize
    Γ2 = @ncon((v, Γ1, u), 
        ([-1, 1], [1, -2, 2], [2, -3])
    )
    Γ2 = permute(Γ2, (1, 2), (3,))

    psi_canonical = Γ2 * λ2

    # truncation
    truncation_tensor = isometry(ℝ^chi, ℝ^target_chi)
    @tensor psi_truncated[-1, -2; -3] :=  truncation_tensor[1, -1] * psi_canonical[1, -2, 2] * truncation_tensor[2, -3]
    #psi_truncated = permute(psi_truncated, (1, 2), (3,))

    # estimated scale of error
    if target_chi == chi
        error_scale = 0
    else
        error_scale = λ2[target_chi+1, target_chi+1]
    end
    return psi_truncated, sqrt(real(wr)), error_scale
end
#@VSCodeServer.run idmrg(psi, 7)
psi = TensorMap(rand, Float64, ℝ^8*ℝ^2, ℝ^8)
psi_1, _, _ = idmrg(psi, 7)
psi_2, _, _ = idmrg(psi_1, 7)
lop = transf_mat(psi_1, psi_1)
lop2 = transf_mat(psi_2, psi_2)
w2, v2 = eigsolve(lop2, Tensor(rand, Float64, ℝ^7*ℝ^7), 1)
w2, v2 = w2[1], v2[1]
v2 = permute(v2, (1, ), (2, ))
v2 = v2 / v2[1,1]

w1, v1 = eigsolve(lop, Tensor(rand, Float64, ℝ^8*ℝ^8), 1)
w1, v1 = w1[1], v1[1]
v1 = permute(v1, (1, ), (2,))
v1
v1[1,1]
v1[2,2]
v1[3,3]
v1[4,4]
v1
svd(v1)


v1 = v1 / v1[1,1]
v1 - v1'




psi
Tpsi=act(T, psi)
psi, _, _ = idmrg(Tpsi, 8)

ovlp(Tpsi, psi) * ovlp(psi, Tpsi) / (ovlp(Tpsi, Tpsi) * ovlp(psi, psi))

psi = TensorMap(rand, Float64, ℝ^8*ℝ^2, ℝ^8)
for ix in 1:10
    Tpsi = act(T, psi)
    psi, norm_psi, error = idmrg(Tpsi, 8)
    F_value = ovlp(psi, act(T, psi)) / ovlp(psi, psi)
    println(ix, ' ', log(F_value |> norm), ' ', norm_psi, ' ', error)
end
Tpsi = act(T, psi)
idmrg(Tpsi, 8)


# tests for idmrg
ovlp(psi_new, psi) * ovlp(psi, psi_new) / ovlp(psi, psi)

@tensor right_contr[:] := Γ2'[3, -2, 4] * λ2'[2, 3] * λ2[1, 2] * Γ2[-1, 4, 1]
right_contr = permute(right_contr, (1,), (2,))
right_contr ≈ id(ℝ^8)

function mps_plus(psi::TensorMap{CartesianSpace, 2, 1}, phi::TensorMap{CartesianSpace, 2, 1})
    chi_psi, chi_phi = dim(domain(psi)), dim(domain(phi))
    chi = chi_psi + chi_phi

    embedder1 = isometry(ℝ^chi, ℝ^chi_psi)
    embedder2 = TensorMap(zeros, Float64, ℝ^chi, ℝ^chi_phi)
    embedder2.data[chi_psi+1:chi, :] = Matrix{Float64}(I, chi_phi, chi_phi)

    @tensor psi_plus_phi[-1, -2; -3] := embedder1[-1, 1] * psi[1, -2, 2] * embedder1[-3, 2] +
             embedder2[-1, 1] * phi[1, -2, 2] * embedder2[-3, 2]

    return psi_plus_phi
end

psi = TensorMap(rand, Float64, ℝ^4*ℝ^2, ℝ^4)
phi = TensorMap(rand, Float64, ℝ^4*ℝ^2, ℝ^4)
psi_add_phi = mps_plus(psi, phi)
psi_add_phi_arr = reshape(psi_add_phi.data, (8, 2, 8))
psi_add_phi_arr[:, 1, :]

psi = TensorMap(rand, Float64, ℝ^2*ℝ^2, ℝ^2)
for ix in 1:20
    Tpsi = act(T, psi)
    Tpsi, _, _ = idmrg(Tpsi, 2) # normalize Tpsi
    psi, norm_psi, error = idmrg(mps_plus(Tpsi, psi), 2)
    F_value = ovlp(psi, act(T, psi)) / ovlp(psi, psi)
    println(" --- ")
    println(ix, ' ', log(F_value |> norm), ' ', norm_psi, ' ', error)
    println(" --- ")
end

psi = TensorMap(rand, Float64, ℝ^8*ℝ^2, ℝ^8)
Tpsi = act(T, psi)

Tpsi1, norm_Tpsi, em_diag,rror = idmrg(Tpsi, 16)
