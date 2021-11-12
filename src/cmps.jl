struct cmps
    Q::TensorMap{ComplexSpace, 1, 1}
    R::TensorMap{ComplexSpace, 2, 1}
end

function cmps(f, chi::Integer, d::Integer)
    Q = TensorMap(f, ComplexF64, ℂ^chi, ℂ^chi)
    R = TensorMap(f, ComplexF64, ℂ^chi*ℂ^d, ℂ^chi)
    return cmps(Q, R)
end

@inline get_chi(psi::cmps) = get_chi(psi.R)
@inline get_d(psi::cmps) = get_d(psi.R)

function transf_mat(psi::cmps, phi::cmps)
    function lop(v::TensorMap{ComplexSpace, 1, 1})
        @tensor Tv[-1; -2] := v[-1, 1] * psi.Q'[1, -2] + 
                            phi.Q[-1, 1] * v[1, -2] + 
                            phi.R[-1, 3, 1] * v[1, 2] * psi.R'[2, -2, 3]
        return Tv
    end
    return lop
end

function transf_mat_T(psi::cmps, phi::cmps)
    function lop_T(v::TensorMap{ComplexSpace, 1, 1})
        @tensor Tv[-1; -2] := v[-1, 1] * psi.Q[1, -2] + 
                            phi.Q'[-1, 1] * v[1, -2] + 
                            phi.R'[-1, 1, 3] * psi.R[2, 3, -2] * v[1, 2]
        return Tv
    end
    return lop_T
end

function left_canonical(psi::cmps)
    chi = get_chi(psi)

    # solve the fixed point equation
    vl = TensorMap(rand, ComplexF64, ℂ^chi, ℂ^chi)
    lop_T = transf_mat_T(psi, psi)
    w, vl = eigsolve(lop_T, vl, 1)
    w = w[1]
    vl = vl[1]

    # obtain gauge transformation matrix X
    _, s, u = tsvd(vl)
    X = sqrt(s) * u
    Xinv = u' * sqrt(inv(s))

    # update Q and R 
    Q = X * psi.Q * Xinv - 0.5 * w * id(ℂ^chi) # normalized
    @tensor R[-1, -2; -3] := X[-1, 1] * psi.R[1, -2, 2] * Xinv[2, -3]

    return cmps(Q, R)
end

function right_canonical(psi::cmps)
    chi = get_chi(psi)

    # solve the fixed point equation
    vr = TensorMap(rand, ComplexF64, ℂ^chi, ℂ^chi)
    lop = transf_mat(psi, psi)
    w, vr = eigsolve(lop, vr, 1)
    w = w[1]
    vr = vr[1]

    # obtain gauge transformation matrix Yinv
    _, s, u = tsvd(vr)
    Y = sqrt(s) * u
    Yinv = u' * sqrt(inv(s))

    # update Q and R 
    Q = Yinv' * psi.Q * Y' - 0.5 * w * id(ℂ^chi)
    @tensor R[-1, -2; -3] := Yinv'[-1, 1] * psi.R[1, -2, 2] * Y'[2, -3]
    
    return cmps(Q, R)
end