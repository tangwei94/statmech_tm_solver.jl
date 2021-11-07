function act(T::TensorMap{ComplexSpace, 2, 2}, psi::TensorMap{ComplexSpace, 2, 1})
    chi = dim(psi.dom)
    fuse_tensor = isomorphism(ℂ^(chi*2), ℂ^2*ℂ^chi)

    #Tpsi = @ncon((T, psi, fuse_tensor, fuse_tensor'), ([3, -2, 2, 5], [1, 2, 4], [-1, 3, 1], [5, 4, -3]))
    #Tpsi = permute(Tpsi, (1, 2), (3,))
    @tensor Tpsi[-1, -2; -3] := T[3, -2, 2, 5] * psi[1, 2, 4] * fuse_tensor[-1, 3, 1] * fuse_tensor'[5, 4, -3]
    return Tpsi
end

function rrule(::typeof(act), T::TensorMap{ComplexSpace, 2, 2}, psi::TensorMap{ComplexSpace, 2, 1})
    chi = dim(psi.dom)
    fuse_tensor = isomorphism(ℂ^(chi*2), ℂ^2*ℂ^chi)

    @tensor Tpsi[-1, -2; -3] := T[3, -2, 2, 5] * psi[1, 2, 4] * fuse_tensor[-1, 3, 1] * fuse_tensor'[5, 4, -3]

    fwd = Tpsi
    function act_pushback(f̄wd)
        #psi_pushback = @ncon((f̄wd, fuse_tensor, fuse_tensor', T), 
        #    ([1,4,2], [1, 3, -1], [5, -3, 2], [3, 4, -2, 5])
        #)
        #psi_pushback = permute(psi_pushback, (1, 2), (3,))
        @tensor psi_pushback[-1, -2; -3] := f̄wd[1,4,2] * fuse_tensor[1,3,-1] * fuse_tensor'[5,-3,2] * T[3,4,-2,5]
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
        @tensor psi_pushback[-2, -3; -1] := conj(vl'[-2, 1]) * conj(phi[1, -3, 2]) * conj(vr[2, -1]) * conj(f̄wd)
        @tensor phi_pushback[-1, -2; -3] := vl'[1, -1] * psi'[2, 1, -2] * vr[-3, 2] * f̄wd 
        return NoTangent(), psi_pushback, phi_pushback
    end
    return fwd, ovlp_pushback
end

function nonherm_cost_func(T::TensorMap{ComplexSpace, 2, 2}, psi_arr::Array{ComplexF64, 3})
    psi = arr_to_TensorMap(psi_arr)
    Tpsi = act(T, psi) 

    up = ovlp(Tpsi, Tpsi) * ovlp(psi, psi)
    dn = ovlp(psi, Tpsi) * ovlp(Tpsi, psi)

    return log(real(up / dn))
end