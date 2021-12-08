struct cmps
    Q::TensorMap{ComplexSpace, 1, 1}
    R::TensorMap{ComplexSpace, 2, 1}
end

function cmps(f, chi::Integer, d::Integer)
    Q = TensorMap(f, ComplexF64, ℂ^chi, ℂ^chi)
    R = TensorMap(f, ComplexF64, ℂ^chi*ℂ^d, ℂ^chi)
    return cmps(Q, R)
end

+(psi::cmps, phi::cmps) = cmps(psi.Q + phi.Q, psi.R + phi.R)
-(psi::cmps, phi::cmps) = cmps(psi.Q - phi.Q, psi.R - phi.R)
*(psi::cmps, x::Number) = cmps(psi.Q * x, psi.R * x)
*(x::Number, psi::cmps) = cmps(psi.Q * x, psi.R * x)
similar(psi::cmps) = cmps(similar(psi.Q), similar(psi.R))

length(iter::cmps) = length(iter.Q.data) + length(iter.R.data)
iterate(iter::cmps) = (iter.Q[1], 1)
function iterate(iter::cmps, state)
    next_state = state + 1
    len_Q = length(iter.Q.data)
    len_R = length(iter.R.data)
    if next_state <= len_Q
        next_elem = iter.Q[next_state]
    elseif next_state <= len_Q + len_R 
        next_elem = iter.R[next_state - len_Q]
    else
        return nothing 
    end
    return (next_elem, next_state)
end
function getindex(psi::cmps, ix::Integer)
    len_Q = length(psi.Q.data)
    len_R = length(psi.R.data)
    if ix <= len_Q && ix > 0
        return psi.Q[ix]
    elseif ix <= len_Q + len_R
        return psi.Q[ix]
    else 
        throw(BoundsError())
    end
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

"""
    leftorth(A::cmps) -> AL, Tϵ

    QR decomposition of cMPS local tensor. 

    To avoid confusion (as we have used Q, R to denote the matrices inside the cMPS local tensor), 
    we use notations A = AL T, where A is the input cMPS local tensor, AL is the left-orthogonalized cMPS local tensor (AL^{\\dagger} AL = I), and T = I + ϵ Tϵ is the upper triangular matrix. 

    Returns AL and Tϵ.
"""
function leftorth(A::cmps)
    chi = get_chi(A)

    # calculate Tϵ
    Tϵ = A.Q + A.Q' + A.R' * A.R
    Tϵ_arr = triu(Tϵ.data) - 0.5 * Diagonal(Tϵ.data)
    Tϵ = TensorMap(Tϵ_arr, ℂ^chi, ℂ^chi)

    # calculate QL, and RL
    QL = A.Q - Tϵ
    RL = A.R 

    return cmps(QL, RL), Tϵ 
end

"""
    left_canonical(psi::cmps)

    Convert the input cmps into the left-canonical form. 
    Return the gauge transformation matrix X and the left-canonicalized cmps. 
"""
function left_canonical(psi::cmps)
    chi = get_chi(psi)

    # solve the fixed point equation
    vl = TensorMap(rand, ComplexF64, ℂ^chi, ℂ^chi)
    lop_T = transf_mat_T(psi, psi)
    w, vl = eigsolve(lop_T, vl, 1, :LR)
    w = w[1]
    vl = vl[1]

    # obtain gauge transformation matrix X
    _, s, u = tsvd(vl)
    X = sqrt(s) * u
    Xinv = u' * sqrt(inv(s))

    # update Q and R 
    Q = X * psi.Q * Xinv - 0.5 * w * id(ℂ^chi) # normalized
    @tensor R[-1, -2; -3] := X[-1, 1] * psi.R[1, -2, 2] * Xinv[2, -3]

    return X, cmps(Q, R)
end

"""
    right_canonical(psi::cmps)

    Convert the input cmps into the right-canonical form. 
    Return the gauge transformation matrix Y and the right-canonicalized cmps. 
"""
function right_canonical(psi::cmps)
    chi = get_chi(psi)

    # solve the fixed point equation
    vr = TensorMap(rand, ComplexF64, ℂ^chi, ℂ^chi)
    lop = transf_mat(psi, psi)
    w, vr = eigsolve(lop, vr, 1, :LR)
    w = w[1]
    vr = vr[1]

    # obtain gauge transformation matrix Yinv
    _, s, u = tsvd(vr)
    Y = sqrt(s) * u
    Yinv = u' * sqrt(inv(s))

    # update Q and R 
    Q = Yinv' * psi.Q * Y' - 0.5 * w * id(ℂ^chi)
    @tensor R[-1, -2; -3] := Yinv'[-1, 1] * psi.R[1, -2, 2] * Y'[2, -3]
    
    return Y, cmps(Q, R)
end

function entanglement_spectrum(psi::cmps)
    _, psi = left_canonical(psi)
    Y, _ = right_canonical(psi)
    _, s, _ = tsvd(Y')
    s = diag(s.data) .^ 2
    return s ./ sum(s)
end

function entanglement_entropy(psi::cmps)
    s = entanglement_spectrum(psi)
    return -sum(s .* log.(s))
end

function expand(psi::cmps, chi::Integer, perturb::Float64=1e-3)
    chi0, d = get_chi(psi), get_d(psi)
    if chi <= chi0
        @warn "chi not larger than current bond D, not expanded "
        return psi
    end

    Q_arr = perturb*rand(ComplexF64, chi, chi)
    R_arr = perturb*rand(ComplexF64, chi, d, chi)
    Q_arr[1:chi0, 1:chi0] += toarray(psi.Q)
    R_arr[1:chi0, :, 1:chi0] += toarray(psi.R)
    Q = TensorMap(Q_arr, ℂ^chi, ℂ^chi)
    R = arr_to_TensorMap(R_arr)

    return cmps(Q, R)
end

"""
    act(op::cmpo, psi::cmps)

    Act a cmpo to a cmps. 
    The tensor structure of the new cmps will not be kept.
"""
function act(T::cmpo, psi::cmps)
    chi_cmpo, chi_psi = get_phy(T), get_chi(psi)
    chi_tot = chi_cmpo * chi_psi
    t_fuse = isomorphism(ℂ^chi_tot, ℂ^chi_cmpo*ℂ^chi_psi)

    @tensor Q[-1; -2] := t_fuse[-1, 1, 3] * T.Q[1, 2] * t_fuse'[2, 3, -2] +
                         t_fuse[-1, 3, 1] * psi.Q[1, 2] * t_fuse'[3, 2, -2] +
                         t_fuse[-1, 1, 2] * T.L'[1, 4, 3] * T.R[2, 3, 5] * t_fuse'[4, 5, -2]
    @tensor R[-1, -2; -3] := t_fuse[-1, 1, 3] * T.R[1, -2, 2] * t_fuse'[2, 3, -3]

    return cmps(Q, R)
end

"""
    rrule(::typeof(act), T::cmpo, psi::cmps)

    Reverse diff rule for `act(op::cmpo, psi::cmps)`
"""
function rrule(::typeof(act), T::cmpo, psi::cmps)
    
end

""" 
    log_ovlp(phi::cmps, psi::cmps, L::Real)

    Caculate the log of overlap for two finite uniform cmps `phi` and `psi`. 
    `L` is the length of the uniform cmps. 
    """
function log_ovlp(phi::cmps, psi::cmps, L::Real)
    Id_phi = id(ℂ^get_chi(phi))
    Id_psi = id(ℂ^get_chi(psi))
    @tensor t_trans[-1, -2; -3, -4] := phi.Q'[-3, -1] * Id_psi[-2, -4] + 
                                       Id_phi'[-3, -1] * psi.Q[-2, -4] + 
                                       phi.R'[-3, -1, 1] * psi.R[-2, 1, -4]
    w, _ = eig(t_trans)
    w = diag(w.data)
    return logsumexp(L*w)
end

"""
    rrule(::typeof(log_ovlp), phi::cmps, psi::cmps, L::Real)

    Reverse diff rule for `log_ovlp(phi::cmps, psi::cmps, L::Real)`
"""
function rrule(::typeof(log_ovlp), phi::cmps, psi::cmps, L::Real)

end
