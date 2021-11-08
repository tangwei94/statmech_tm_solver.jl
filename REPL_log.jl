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

psi = TensorMap(rand, ComplexF64, ℂ^8*ℂ^2, ℂ^8)
T

Tpsi = act(T, psi)
_, phi = iTEBD_truncate(Tpsi, 8)

pseudo_ovlp(phi, Tpsi, psi, 4) 
pseudo_ovlp(phi, Tpsi, psi, 8) 
pseudo_ovlp(phi, Tpsi, psi, 16) 
pseudo_ovlp(phi, Tpsi, psi, 32) 
ovlp(phi, mps_add(Tpsi, psi)) |> log

ovlp(phi, Tpsi)
ovlp(phi, psi)