function act(T::TensorMap{ComplexSpace, 2, 2}, psi::TensorMap{ComplexSpace, 2, 1})
    chi = get_chi(psi)
    d = dims(domain(T))[2]
    fuse_tensor = isomorphism(ℂ^(chi*d), ℂ^d*ℂ^chi)

    @tensor Tpsi[-1, -2; -3] := T[3, -2, 2, 5] * psi[1, 2, 4] * fuse_tensor[-1, 3, 1] * fuse_tensor'[5, 4, -3]
    return Tpsi
end

function rrule(::typeof(act), T::TensorMap{ComplexSpace, 2, 2}, psi::TensorMap{ComplexSpace, 2, 1})
    chi = get_chi(psi)
    d = dims(domain(T))[2]
    fuse_tensor = isomorphism(ℂ^(chi*d), ℂ^d*ℂ^chi)

    @tensor Tpsi[-1, -2; -3] := T[3, -2, 2, 5] * psi[1, 2, 4] * fuse_tensor[-1, 3, 1] * fuse_tensor'[5, 4, -3]

    fwd = Tpsi
    function act_pushback(f̄wd)
        @tensor psi_pushback[-1, -2; -3] := f̄wd[1,4,2] * conj(fuse_tensor[1,3,-1]) * conj(fuse_tensor'[5,-3,2]) * conj(T[3,4,-2,5])
        return NoTangent(), NoTangent(), psi_pushback
    end
    return fwd, act_pushback
end

function transf_mat(psi::TensorMap{ComplexSpace, 2, 1}, phi::TensorMap{ComplexSpace, 2, 1})
    function lop(v::TensorMap{ComplexSpace, 1, 1})
        @tensor result[-2; -1] := psi'[1, -1, 3] * phi[-2, 3, 2] * v[2, 1]
        return result
    end
    return lop
end

function transf_mat_T(psi::TensorMap{ComplexSpace, 2, 1}, phi::TensorMap{ComplexSpace, 2, 1})
    function lop_T(v::TensorMap{ComplexSpace, 1, 1})
        @tensor result[-2; -1] := psi[1, 3, -1] * phi'[-2, 2, 3] * v[2, 1]
        return result
    end
    return lop_T
end

function transf_mat(A::TensorMap{ComplexSpace, 2, 1}, O::TensorMap{ComplexSpace, 2, 2}, B::TensorMap{ComplexSpace, 2, 1})
    function lop(v::TensorMap{ComplexSpace, 2, 1})
        @tensor v1[-1, -2; -3] := A[-1, 5, 1] * O[-2, 4, 5, 3] * B'[2, -3, 4] * v[1, 3, 2]
        return v1
    end
    return lop
end

function transf_mat_T(A::TensorMap{ComplexSpace, 2, 1}, O::TensorMap{ComplexSpace, 2, 2}, B::TensorMap{ComplexSpace, 2, 1})
    function lop_T(v::TensorMap{ComplexSpace, 2, 1})
        @tensor v1[-1, -2; -3] := A'[-1, 1, 5] * O'[5, -2, 3, 4] * B[2, 4, -3] * v[1, 3, 2]
        return v1
    end
    return lop_T
end

function ovlp(psi::TensorMap{ComplexSpace, 2, 1}, phi::TensorMap{ComplexSpace, 2, 1})
    chi_psi, chi_phi = get_chi(psi), get_chi(phi)
    v0 = TensorMap(rand, ComplexF64, ℂ^chi_phi, ℂ^chi_psi)
    w, _ = eigsolve(transf_mat(psi, phi), v0, 1)
    return w[1]
end

function rrule(::typeof(ovlp), psi::TensorMap{ComplexSpace, 2, 1}, phi::TensorMap{ComplexSpace, 2, 1})
    chi_psi, chi_phi = get_chi(psi), get_chi(phi)
    v0 = TensorMap(rand, ComplexF64, ℂ^chi_phi, ℂ^chi_psi)
    wr, vr = eigsolve(transf_mat(psi, phi), v0, 1)
    _, vl = eigsolve(transf_mat_T(psi, phi), v0, 1)
    fwd = wr[1]
    vl, vr = vl[1], vr[1]
    vr = vr / dot(vl, vr)

    function ovlp_pushback(f̄wd)
        @tensor psi_pushback[-2, -3; -1] := vl'[-2, 1] * phi[1, -3, 2] * vr[2, -1] * conj(f̄wd)
        @tensor phi_pushback[-1, -2; -3] := conj(vl'[1, -1]) * conj(psi'[2, 1, -2]) * conj(vr[-3, 2]) * f̄wd 
        return NoTangent(), psi_pushback, phi_pushback
    end
    return fwd, ovlp_pushback
end

function nonherm_cost_func(T::TensorMap{ComplexSpace, 2, 2}, psi::TensorMap{ComplexSpace, 2, 1})
    Tpsi = act(T, psi) 

    up = ovlp(Tpsi, Tpsi) * ovlp(psi, psi)
    dn = ovlp(psi, Tpsi) * ovlp(Tpsi, psi)

    return log(norm(up / dn))
end

function lambda_gamma(psi::TensorMap{ComplexSpace, 2, 1})
    chi = get_chi(psi)

    lop = transf_mat(psi, psi)
    lopT = transf_mat_T(psi, psi)

    ρ0 = TensorMap(rand, ComplexF64, ℂ^chi, ℂ^chi)
    wr, ρr = eigsolve(lop, ρ0, 1)
    _, ρl = eigsolve(lopT, ρ0, 1)
    ρl, ρr = ρl[1], ρr[1]
    norm_psi = wr[1] |> norm |> sqrt
    psi = psi / norm_psi

    _, s, v = tsvd(ρr)
    Y = sqrt(s) * v
    Yinv = v' * inv(sqrt(s)) 

    _, s, v = tsvd(ρl')
    X = sqrt(s) * v
    Xinv = v' * inv(sqrt(s)) 

    u, Λ, v = tsvd(X * Y')
    @tensor Γ[-1, -2; -3] := 
        v[-1, 1] * Yinv'[1, 3] * psi[3, -2, 4] * Xinv[4, 2] * u[2, -3]
    return norm_psi, Γ, Λ
end

function left_canonical(psi::TensorMap{ComplexSpace, 2, 1})

    norm_psi, Γ, Λ = lambda_gamma(psi)
    @tensor psi_L[-1, -2; -3] := Λ[-1, 1] * Γ[1, -2, -3]
    return norm_psi, psi_L

end

function iTEBD_truncate(psi::TensorMap{ComplexSpace, 2, 1}, chi::Integer)
    norm_psi, Γ, Λ = lambda_gamma(psi)
    chi0 = get_chi(psi)

    P = isometry(ℂ^chi0, ℂ^chi)
    @tensor psi_truncated[-1, -2; -3] :=
        P'[-1, 1] * Λ[1, 2] * Γ[2, -2, 3] * P[3, -3]
    return norm_psi, psi_truncated
end

function ln_fidelity(psi::TensorMap{ComplexSpace, 2, 1}, phi::TensorMap{ComplexSpace, 2, 1})
    up = ovlp(psi, phi) * ovlp(phi, psi)
    dn = ovlp(psi, psi) * ovlp(phi, phi)
    return (up/dn) |> norm |> log
end

function free_energy(T::TensorMap{ComplexSpace, 2, 2}, psi::TensorMap{ComplexSpace, 2, 1})
    Tpsi = act(T, psi)
    up = ovlp(psi, Tpsi)
    dn = ovlp(psi, psi)
    return (up/dn) |> norm |> log
end

function mps_add(psi::TensorMap{ComplexSpace, 2, 1}, phi::TensorMap{ComplexSpace, 2, 1})
    # https://github.com/Jutho/TensorKit.jl/issues/54
    # actually doesn't work for infinite MPS 
    chi_psi, chi_phi = get_chi(psi), get_chi(phi)
    chi = chi_psi + chi_phi

    embedder1 = isometry(ℂ^chi, ℂ^chi_psi)
    embedder2 = TensorMap(zeros, ComplexF64, ℂ^chi, ℂ^chi_phi)
    embedder2.data[chi_psi+1:chi, :] = Matrix{ComplexF64}(I, chi_phi, chi_phi)

    @tensor psi_plus_phi[-1, -2; -3] := embedder1[-1, 1] * psi[1, -2, 2] * embedder1'[2, -3] +
             embedder2[-1, 1] * phi[1, -2, 2] * embedder2'[2, -3]

    return psi_plus_phi
end

function right_canonical_QR(psi::TensorMap{ComplexSpace, 2, 1}, tol::Float64=1e-15)

    chi = get_chi(psi)
    L0 = id(ℂ^chi)

    L, Q = rightorth(permute(psi, (1, ), (2, 3)))
    psi_R = permute(Q, (1, 2), (3, ))
    L = L / norm(L)
    δ = norm(L - L0)
    L0 = L

    ix= 0
    while δ > tol && ix < 200
        if ix >= 20 && ix % 10 == 0 
            lop = transf_mat(psi, psi_R)
            _, vr = eigsolve(lop, L0, 1; tol=max(tol, δ/10))
            L = vr[1]' 
            #print("--> ")
        end

        L, Q = rightorth(permute(psi * L, (1, ), (2, 3)))
        psi_R = permute(Q, (1, 2), (3, ))
        L = L / norm(L)

        δ = norm(L-L0)
        L0 = L

        #println(δ) 
        ix += 1
    end
    
    #println(ix, " iterations")
    δ > tol && @warn "right_canonical_QR failed to converge. δ: $δ , tol: $tol"

    return L0', psi_R
end

function left_canonical_QR(psi::TensorMap{ComplexSpace, 2, 1}, tol::Float64=1e-15)
    chi = get_chi(psi)
    R0 = id(ℂ^chi)

    psi_L, R = leftorth(psi)
    R = R / norm(R)
    δ = norm(R - R0)
    R0 = R

    ix = 0
    while δ > tol && ix < 200
        if ix >= 20 && ix % 10 == 0 
            lop_T = transf_mat_T(psi, psi_L)
            _, vl = eigsolve(lop_T, R0, 1; tol=max(tol, δ/10))
            R = vl[1]
            #print("--> ")
        end

        @tensor psi_tmp[-1, -2; -3] := R[-1, 1] * psi[1, -2, -3]
        psi_L, R = leftorth(psi_tmp)
        R = R / norm(R)

        δ = norm(R - R0)
        R0 = R

        #println(δ) 
        ix += 1
    end

    #println(ix, " iterations")
    δ > tol && @warn "left_canonical_QR failed to converge. δ: $δ , tol: $tol"

    return R0, psi_L
end

function entanglement_spectrum(psi::TensorMap{ComplexSpace, 2, 1})
    _, psi = left_canonical_QR(psi)
    Y, _ = right_canonical_QR(psi)
    _, s, _ = tsvd(Y')
    s = diag(s.data) .^ 2
    return s ./ sum(s)
end

function entanglement_entropy(psi::TensorMap{ComplexSpace, 2, 1})
    s = entanglement_spectrum(psi)
    return -sum(s .* log.(s))
end

function expand(psi::TensorMap{ComplexSpace, 2, 1}, chi::Integer, perturb::Float64=1e-3)
    chi0, d = get_chi(psi), get_d(psi)
    if chi <= chi0
        @warn "chi not larger than current bond D, not expanded "
        return psi
    end

    phi_arr = perturb*rand(ComplexF64, chi, d, chi)
    phi_arr[1:chi0, :, 1:chi0] += toarray(psi)
    phi = arr_to_TensorMap(phi_arr)

    return phi
end

function tangent_map_tn(O::TensorMap{ComplexSpace, 2, 2}, AL::TensorMap{ComplexSpace, 2, 1}, AR::TensorMap{ComplexSpace, 2, 1})
    lop_R = transf_mat(AR, O, AR)
    lop_L = transf_mat_T(AL, O, AL)
    chi_mps = get_chi(AL)
    chi_mpo = dims(domain(O))[2]

    ER = TensorMap(rand, ComplexF64, ℂ^chi_mps*ℂ^chi_mpo, ℂ^chi_mps)
    EL = TensorMap(rand, ComplexF64, ℂ^chi_mps*ℂ^chi_mpo, ℂ^chi_mps)
    _, ER = eigsolve(lop_R, ER, 1)
    _, EL = eigsolve(lop_L, EL, 1)
    EL, ER = EL[1], ER[1]
    
    @tensor norm = ER[1,2,3]*EL'[3,1,2]
    ER = ER / norm

    @tensor map_AC[-1, -2, -3; -4, -5, -6] := EL'[-1, -4, 1] * O[1, -2, -5, 2] * ER[-6, 2, -3]
    @tensor map_C[-1, -2; -3, -4] := EL'[-1, -3, 1] * ER[-4, 1, -2]

    return map_AC, map_C
end

function tangent_map(O::TensorMap{ComplexSpace, 2, 2}, AL::TensorMap{ComplexSpace, 2, 1}, AR::TensorMap{ComplexSpace, 2, 1})
    lop_R = transf_mat(AR, O, AR)
    lop_L = transf_mat_T(AL, O, AL)
    chi_mps = get_chi(AL)
    chi_mpo = dims(domain(O))[2]

    ER = TensorMap(rand, ComplexF64, ℂ^chi_mps*ℂ^chi_mpo, ℂ^chi_mps)
    EL = TensorMap(rand, ComplexF64, ℂ^chi_mps*ℂ^chi_mpo, ℂ^chi_mps)
    _, ER = eigsolve(lop_R, ER, 1)
    _, EL = eigsolve(lop_L, EL, 1)
    EL, ER = EL[1], ER[1]
    
    @tensor norm = ER[1,2,3]*EL'[3,1,2]
    ER = ER / norm

    function map_AC(AC::TensorMap{ComplexSpace, 2, 1})
        @tensor updated_AC[-1, -2; -3] := AC[1, 3, 2] * EL'[-1, 1, 4] * ER[2, 5, -3] * O[4, -2, 3, 5] 
        return updated_AC
    end
    function map_C(C::TensorMap{ComplexSpace, 1, 1})
        @tensor updated_C[-1; -2] := C[1, 2] * EL'[-1, 1, 3] * ER[2, 3, -2] 
        return updated_C
    end
    return map_AC, map_C
end

function calculate_ALR(AC::TensorMap{ComplexSpace, 2, 1}, C::TensorMap{ComplexSpace, 1, 1})
    UAC_l, _ = leftorth(AC; alg=Polar())
    UC_l, _ = leftorth(C; alg=Polar())

    _, UAC_r = rightorth(permute(AC, (1,), (2,3)); alg=Polar())
    _, UC_r = rightorth(C; alg=Polar())

    AL = UAC_l * UC_l'
    AR = permute(UC_r' * UAC_r, (1, 2), (3,))

    return AL, AR    
end