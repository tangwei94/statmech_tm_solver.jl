# put unsuccessful code here
# just in case

# from imps.jl
# imps uninjectivity issue. tried to regulize it with finite power order
function pseudo_ovlp(psi::TensorMap{ComplexSpace, 2, 1}, phi1::TensorMap{ComplexSpace, 2, 1}, phi2::TensorMap{ComplexSpace, 2, 1}, power_order::Integer)
    ln_ovlp1 = log(ovlp(psi, phi1))
    ln_ovlp2 = log(ovlp(psi, phi2))
    ln_ovlp_tot = ln_ovlp1 + log1p(exp(power_order*(ln_ovlp2-ln_ovlp1))) / power_order
    return ln_ovlp_tot

end

function variational_truncate(phi::TensorMap{ComplexSpace, 2, 1}, chi::Integer)
    _, phi = left_canonical(phi)

    _, psi = iTEBD_truncate(phi, chi)
    psi_arr = toarray(psi)

    function _f(_arr)
        _psi = arr_to_TensorMap(_arr)
        up = norm(ovlp(_psi, phi))^2  
        dn = ovlp(_psi, _psi)
        return -(up/dn) |> real 
    end
    function _g!(_grad, _arr)
        _grad .= gradient(_f, _arr)[1]
    end

    res_f = optimize(_f, _g!, psi_arr, LBFGS(), Optim.Options(show_trace=false, iterations=100))
    psi = Optim.minimizer(res_f) |> arr_to_TensorMap

    return psi
end

# from cmps.jl
# periodic uniform cMPS compression issue. tried to construct the projector into the cMPS tangent space. not working. 
function C_matrix(D::TensorMap{ComplexSpace, 1, 1}, L::Real)
    Cmat = similar(D)
    w_vec = diag(D.data)
    N = length(w_vec)
    for ix in 1:N
        for iy in 1:N
            if w_vec[ix] ≈ w_vec[iy]
                Cmat[ix, iy] = exp(w_vec[ix])
            else
                Cmat[ix, iy] = (exp(w_vec[ix] * L) - exp(w_vec[iy] * L)) / (w_vec[ix] - w_vec[iy])
            end
        end
    end
    return Cmat
end

function delta_tensor(chi2::Integer)
    δ = TensorMap(zeros, ComplexF64, ℂ^chi2*ℂ^chi2, ℂ^chi2)
    for ix in 1:chi2
        δ[ix, ix, ix] = 1
    end
    return δ
end

function company_tensor(psi::cmps)
    A = convert_to_array(psi)
    chi = get_chi(psi)
    A[:, 1, :] = Matrix{ComplexF64}(I, chi, chi)
    return convert_to_tensormap(A, 2)
end

"""
    gram_matrix(psi::cmps, L::Real)

    calculate the gram matrix. 
"""
function gram_matrix(psi::cmps, L::Real)
    chi = get_chi(psi)

    K = K_mat(psi, psi)
    K = permute(K, (2, 1), (4, 3))
    D, VR = eig(K)
    VL = inv(VR)' # left eigenvalues 
   
    VR = permute(VR, (1,), (3, 2))
    VL = permute(VL, (1,), (3, 2))

    δ = delta_tensor(chi^2)
    C = C_matrix(D, L)
    A = company_tensor(psi)
    Iph = id(ℂ^(get_d(psi)+1))
    Iph[1,1] = 0
    expD = exp(D*L)

    @tensor gram[-1, -2, -3; -4, -5, -6] := 
        δ[3, 2, 1] * VL'[1, 7, -5] * A'[8, 7, -6] * VR[-4, 4, 8] * VR[9, 2, -3] * A[10, -2, 9] * VL'[5, -1, 10] * δ'[4, 5, 6] * C[6, 3] + 
        VL'[2, -1, -5] * Iph[-2, -6] * VR[-4, 1, -3] * expD[1, 2] 
    return gram
end

"""
    tangent_proj(phi::cmps, psi::cmps, L::Real)

    project cmps `phi` into the tangent space of cmps `psi`. `L` is the length of the cmps. 
"""
function tangent_proj(phi::cmps, psi::cmps, L::Real, tol::Real)

    gram_psi = gram_matrix(psi, L)
    gram_psi_inv = pinv(gram_psi; rtol=tol)
    gram_psi_inv = permute(gram_psi_inv, (6,2,3), (4,5,1))

    K = transf_mat(psi, phi)
    KT = transf_mat_T(psi, phi)
    V0 = TensorMap(rand, ComplexF64, ℂ^get_chi(phi), ℂ^get_chi(psi))

    _, VR = eigsolve(K, V0, 1, :LR)
    VR = VR[1]
    _, VL = eigsolve(KT, V0, 1, :LR)
    VL = VL[1]

    phi_tMap = company_tensor(phi)

    @tensor B_proj[-1, -2; -3] := phi_tMap[1, 5, 2] * VL'[3, 1] * VR[2, 4] * gram_psi_inv[4, -1, -2, 3, 5, -3]

    return B_proj
end