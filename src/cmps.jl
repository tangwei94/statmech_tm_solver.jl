struct cmps
    Q::TensorMap{ComplexSpace, 1, 1}
    R::TensorMap{ComplexSpace, 2, 1}
end

"""
    rrule(::typeof(cmps), Q::TensorMap{ComplexSpace, 1, 1}, R::TensorMap{ComplexSpace, 2, 1})

    rrule for the constructor of `cmps`.
"""
function rrule(::typeof(cmps), Q::TensorMap{ComplexSpace, 1, 1}, R::TensorMap{ComplexSpace, 2, 1})
    function cmps_pushback(f̄wd)
        return NoTangent(), f̄wd.Q, f̄wd.R
    end
    return cmps(Q, R), cmps_pushback
end

"""
    cmps(f, chi::Integer, d::Integer) -> cmps

    generate a cmps with vitual dim `chi` and physical dim `d` using function `f` to generate the parameters.
    Example: `cmps(rand, 2, 4)` 
"""
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

"""
    get_matrices(psi::cmps) -> tuple 

    return the Q and R matrices in cMPS `psi`. 
    Q and R will be in the form of a tuple of `TensorMap`s. 
"""
get_matrices(psi::cmps) = (psi.Q, psi.R)

"""
    rrule(::typeof(get_matrices), psi::cmps)
"""
function rrule(::typeof(get_matrices), psi::cmps)
    fwd = (psi.Q, psi.R) 
    function get_Q_pushback(f̄wd)
        (Q̄, R̄) = f̄wd
        (Q̄ isa ZeroTangent) && (Q̄ = TensorMap(zeros, ComplexF64, space(psi.Q)))
        (R̄ isa ZeroTangent) && (R̄ = TensorMap(zeros, ComplexF64, space(psi.R)))
        p̄si = cmps(Q̄, R̄)
        return NoTangent(), p̄si
    end 
    return fwd, get_Q_pushback
end

"""
    convert_to_cmps(arr::Array{<:Complex, 3})

    Convert a chi*(d+1)*chi array to a cMPS. 
    `chi` is the virtual bond dimension, `d+1` is the physical bond dimension. 
"""
function convert_to_cmps(arr::Array{<:Complex, 3})
    Q = convert_to_tensormap(arr[:, 1, :], 1)
    R = convert_to_tensormap(arr[:, 2:end, :], 2)
    return cmps(Q, R)
end

"""
    convert_to_cmps(arr::TensorMap{ComplexSpace, 2, 1})
    
    Convert a ℂ^chi*ℂ^(d+1) ← ℂ^chi TensorMap to a cMPS. 
"""
function convert_to_cmps(arr::TensorMap{ComplexSpace, 2, 1})
    return convert_to_cmps(convert_to_array(arr))
end

"""
    convert_to_array(psi::cmps)

    Convert a cmps to an array.
"""
function convert_to_array(psi::cmps)
    chi = get_chi(psi)
    Q_arr, R_arr = convert_to_array(psi.Q), convert_to_array(psi.R)
    Q_arr = reshape(Q_arr, (chi, 1, chi))
    arr = cat(Q_arr, R_arr, dims=2)
    return arr
end

"""
    convert_to_tensormap(psi::cmps)

    Convert a cmps to a tensormap.
"""
convert_to_tensormap(psi::cmps) = convert_to_tensormap(convert_to_array(psi), 2)

"""
    transf_mat(psi::cmps, phi::cmps) -> Function

    Obtain the transfer matrix of <psi|phi> as a linear operator.
"""
function transf_mat(psi::cmps, phi::cmps)
    function lop(v::TensorMap{ComplexSpace, 1, 1})
        @tensor Tv[-1; -2] := v[-1, 1] * psi.Q'[1, -2] + 
                            phi.Q[-1, 1] * v[1, -2] + 
                            phi.R[-1, 3, 1] * v[1, 2] * psi.R'[2, -2, 3]
        return Tv
    end
    return lop
end

"""
    transf_mat_T(psi::cmps, phi::cmps) -> Function

    Obtain the Hermtian conjugate of the transfer matrix of <psi|phi> as a linear operator.
"""
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
    right_canonical(psi::cmps) -> Y::Matrix, psiR::cmps

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

"""
    entanglement_spectrum(psi::cmps) -> Vector

    return the entanglement spectrum of an infinite cmps.
"""
function entanglement_spectrum(psi::cmps)
    _, psi = left_canonical(psi)
    Y, _ = right_canonical(psi)
    _, s, _ = tsvd(Y')
    s = diag(s.data) .^ 2
    return s ./ sum(s)
end

"""
    entanglement_entropy(psi::cmps) -> Float64

    return the entanglement entropy.
"""
function entanglement_entropy(psi::cmps)
    s = entanglement_spectrum(psi)
    return -sum(s .* log.(s))
end

"""
    expand(psi::cmps, chi::Integer, perturb::Float64=1e-3) -> cmps

    expand the cmps `psi` to a target bond dimension `chi` by adding small numbers of size `perturb`.
"""
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
    act(op::cmpo, psi::cmps) -> cmps

    Act a cmpo to a cmps. 
"""
function act(T::cmpo, psi::cmps)
    chi_cmpo, chi_psi = get_phy(T), get_chi(psi)
    chi_tot = chi_cmpo * chi_psi
    t_fuse = isomorphism(ℂ^chi_tot, ℂ^chi_cmpo*ℂ^chi_psi)

    @tensor Q[-1; -2] := t_fuse[-1, 1, 3] * T.Q[1, 2] * t_fuse'[2, 3, -2] +
                         t_fuse[-1, 3, 1] * psi.Q[1, 2] * t_fuse'[3, 2, -2] +
                         t_fuse[-1, 1, 2] * T.L'[1, 4, 3] * psi.R[2, 3, 5] * t_fuse'[4, 5, -2]
    @tensor R[-1, -2; -3] := t_fuse[-1, 1, 3] * T.R[1, -2, 2] * t_fuse'[2, 3, -3]

    return cmps(Q, R)
end

"""
    rrule(::typeof(act), T::cmpo, psi::cmps)

    Reverse diff rule for `act(op::cmpo, psi::cmps)`. 
    We only calculate the adjoint of `psi`. 
"""
function rrule(::typeof(act), T::cmpo, psi::cmps)
    chi_cmpo, chi_psi = get_phy(T), get_chi(psi)
    chi_tot = chi_cmpo * chi_psi
    t_fuse = isomorphism(ℂ^chi_tot, ℂ^chi_cmpo*ℂ^chi_psi)
    fwd = act(T, psi)

    function act_pushback(f̄wd)
        @tensor Q̄_psi[-1; -2] := f̄wd.Q[1, 2] * t_fuse'[3, -1, 1] * t_fuse[2, 3, -2]
        @tensor R̄_psi[-1, -2; -3] := f̄wd.Q[1, 2] * t_fuse'[3, -1, 1] * t_fuse[2, 4, -3] * T.L[4, -2, 3] 
        p̄si = cmps(Q̄_psi, R̄_psi)
        return NoTangent(), NoTangent(), p̄si 
    end
    return fwd, act_pushback
end

"""
    K_mat(phi::cmps, psi::cmps) -> Kmat::TensorMap{ComplexSpace, 2, 2}

    calculate the K_mat from two cmpses `phi` and `psi`. order of indices:

        -1 -->--  phi' -->-- -3
                   |
                 1 ^
                   |
        -2 --<--  psi  --<-- -4

"""
function K_mat(phi::cmps, psi::cmps)
    Id_phi = id(ℂ^get_chi(phi))
    Id_psi = id(ℂ^get_chi(psi))
    @tensor Kmat[-1, -2; -3, -4] := phi.Q'[-3, -1] * Id_psi[-2, -4] + 
                                    Id_phi'[-3, -1] * psi.Q[-2, -4] + 
                                    phi.R'[-3, -1, 1] * psi.R[-2, 1, -4]
    return Kmat
end

"""
    rrule(::typeof(K_mat), phi::cmps, psi::cmps) -> fwd::TensorMap{ComplexSpace, 2, 2}, K_mat_pushback::Function

    The reverse rule for function K_mat.  
"""
function rrule(::typeof(K_mat), phi::cmps, psi::cmps)
    Id_phi = id(ℂ^get_chi(phi))
    Id_psi = id(ℂ^get_chi(psi))
    fwd = K_mat(phi, psi) 

    function K_mat_pushback(f̄wd)
        f̄wd = permute(f̄wd, (3, 2), (1, 4))
        @tensor Q̄_phi[-1; -2] := conj(f̄wd[-2, 1, -1, 1])
        @tensor Q̄_psi[-1; -2] := f̄wd[1, -1, 1, -2]
        @tensor R̄_phi[-1, -2; -3] := conj(f̄wd[-3, 1, -1, 2]) * conj(psi.R'[2, 1, -2])
        @tensor R̄_psi[-1, -2; -3] := f̄wd[2, -1, 1, -3] * phi.R[1, -2, 2]
        p̄hi = cmps(Q̄_phi, R̄_phi)
        p̄si = cmps(Q̄_psi, R̄_psi)
        return NoTangent(), p̄hi, p̄si
    end
    return fwd, K_mat_pushback
end

""" 
    log_ovlp(phi::cmps, psi::cmps, beta::Real) -> ComplexF64

    Caculate the log of overlap for two finite uniform cmps `phi` and `psi`. 
    `beta` is the length of the uniform cmps. 
"""
function log_ovlp(phi::cmps, psi::cmps, beta::Real; sym::Bool=false)
    t_trans = convert_to_array(K_mat(phi, psi))
    tot_dim = get_chi(phi) * get_chi(psi)
    t_trans = reshape(t_trans, (tot_dim, tot_dim))
    sym && (t_trans = Hermitian(0.5*(t_trans+t_trans')))
    w = eigvals(t_trans)
    return logsumexp(beta*w)
end

""" 
    convergence_meaure(T::cmpo, ψ::cmps, beta::Real) -> Real
"""
function convergence_measure(T::cmpo, ψ::cmps, beta::Real)
    Tψ = act(T, ψ)
    return -2*log_ovlp(Tψ, ψ, beta; sym=true) + log_ovlp(Tψ, Tψ, beta; sym=true) + log_ovlp(ψ, ψ, beta; sym=true) |> real
end

"""
    finite_env(t::TensorMap{ComplexSpace}, L::Real)

    For a cMPS transfer matrix `t` (see `K_mat`), calculate the environment block for the finite length `L`.
    The input `t` matrix should look like 
    ``` 
        -1 -->--  phi' -->-- -3
                   |
                   ^
                   |
        -2 --<--  psi  --<-- -4
    ```
    This function will calculate exp(t) / tr(exp(t)) by diagonalizing `t`.
"""
function finite_env(t::TensorMap{ComplexSpace}, L::Real)
    W, UR = eig(t)
    UL = inv(UR)

    Wvec = diag(W.data)
    Wvec = Wvec .- logsumexp(Wvec * L) / L #normalize
    expW = convert_to_tensormap(diagm(exp.(Wvec * L)), 1)
   
    return UR * expW * UL
end

"""
    rrule(::typeof(finite_env), t::TensorMap{ComplexSpace}, L::Real)

    Backward rule for `finite_env`.
    See https://math.stackexchange.com/a/3868894/488003 and https://doi.org/10.1006/aama.1995.1017 for the gradient of exp(t)
"""
function rrule(::typeof(finite_env), t::TensorMap{ComplexSpace}, L::Real)
    W, UR = eig(t)
    UL = inv(UR)

    Wvec = diag(W.data)
    Wvec = Wvec .- logsumexp(Wvec * L) / L #normalize
    expW = convert_to_tensormap(diagm(exp.(Wvec * L)), 1)
    fwd = UR * expW * UL

    function finite_env_pushback(f̄wd)
        function coeff(a::Number, b::Number) 
            if a ≈ b
                return L*exp(a*L)
            else 
                return (exp(a*L) - exp(b*L)) / (a - b)
            end
        end
        M = UR' * f̄wd * UL'
        M1 = similar(M)
        copyto!(M1.data, M.data .* coeff.(Wvec', conj.(Wvec)))
        t̄ = UL' * M1 * UR' - L * tr(f̄wd * fwd') * fwd'
        
        return NoTangent(), t̄, NoTangent()
    end 
    return fwd, finite_env_pushback
end


"""
    optimize_conv_meas(T::cmpo, ψ::cmps, beta::Real)

    Optimize the convergence measure by gradient optimization. 
"""
function optimize_conv_meas(T::cmpo, ψ::cmps, beta::Real, Niter::Integer)
    ψarr = convert_to_array(ψ)

    function _f(x::Array{ComplexF64, 3})
        ϕ = convert_to_cmps(x)
        return convergence_measure(T, ϕ, beta)
    end 
    function _g!(gx::Array{ComplexF64, 3}, x::Array{ComplexF64, 3})
        gx .= gradient(_f, x)[1]
    end

    res = optimize(_f, _g!, ψarr, Optim.Options(show_trace=false, iterations=Niter))
    return convert_to_cmps(Optim.minimizer(res))
end


"""
    normalize(psi::cmps, beta::Real)

    Normalize the given cmps `psi`. 
"""
function normalize(psi::cmps, beta::Real)
    log_psi_norm = log_ovlp(psi, psi, beta; sym=true)
    id_psi = id(ℂ^(get_chi(psi)))
    @show log_psi_norm
    Q = psi.Q - log_psi_norm / beta / 2  * id_psi
    return cmps(Q, psi.R)
end

"""
    compress(psi::cmps, chi::Integer, beta::Real) -> Bool, Real, cmps

    Compress the cmps `psi` to the target bond dimension `chi`.
    Only implemented the case for symmetric cmps.
"""
function compress(ψ::cmps, chi::Integer, beta::Real; Niter::Integer=100, tol::Real=1e-12, init=nothing)
    chi_ψ = get_chi(ψ)
    if chi_ψ <= chi
        @warn "target bond dimension $(chi) larger than the current one $(chi_ψ)."
        return true, 0.0, ψ 
    end

    K = K_mat(ψ, ψ)
    K = 0.5*(K + K')
    W, U = eigh(K)

    log_norm_ψ = logsumexp(diag(W.data)*beta) / beta

    if init === nothing
        Wmax = findmax(W.data)[1]
        expK = U * exp((W - Wmax * id(ℂ^chi_ψ^2)) * beta / 2) * U'
        expK = permute(expK, (2, 3), (1, 4))

        _, _, Ua = tsvd(expK, (4, 1, 2), (3,))
        #Ub, Sb, _ = tsvd(expK, (1,), (2, 3, 4))
        #Uc, Sc, _ = tsvd(expK, (2,), (3, 4, 1))
        #_, Sd, Ud = tsvd(expK, (1, 2, 3), (4,))
        Pa_arr = Ua.data[1:chi, :]
        Pa = TensorMap(Pa_arr, ℂ^chi, ℂ^chi_ψ)
        Q = Pa * ψ.Q * Pa' 
        @tensor R[-1, -2; -3] := Pa[-1, 1] * ψ.R[1, -2, 2] * Pa'[2, -3]
        ϕ = cmps(Q, R)
    else
        ϕ = init
    end

    log_fidel = 2*log_ovlp(ϕ, ψ, beta) - log_ovlp(ϕ, ϕ, beta) - log_norm_ψ * beta |> real 

    if log_fidel < -tol || init !== nothing
        function _f(x::Array{ComplexF64, 3})
            ϕx = convert_to_cmps(x)
            return - log_ovlp(ϕx, ψ, beta; sym=true) - log_ovlp(ψ, ϕx, beta; sym=true) + log_ovlp(ϕx, ϕx, beta; sym=true) |> real
        end 
        function _g!(gx::Array{ComplexF64, 3}, x::Array{ComplexF64, 3})
            gx .= gradient(_f, x)[1]
        end

        res = optimize(_f, _g!, convert_to_array(ϕ), LBFGS(), Optim.Options(show_trace=false, iterations=Niter, f_abstol=tol))
        ϕ = convert_to_cmps(Optim.minimizer(res))
    end
    ϕ = normalize(ϕ, beta)

    log_fidel = 2*log_ovlp(ϕ, ψ, beta) - log_norm_ψ * beta |> real
    status = (log_fidel > -tol)
    return status, log_fidel, ϕ
end

"""
    truncation_measure(psi::cmps, psi1::cmps, beta::Real)

    Check the quality of periodic cMPS truncation from the following aspects:
    - fidelity 
    - entanglement spectra
    - TODO. correlation function 
"""
function truncation_check(psi::cmps, psi1::cmps, beta::Real)
    # normalize psi and psi_truncated 
    chi, chi1 = get_chi(psi), get_chi(psi1)
    Q = psi.Q - log_ovlp(psi, psi, beta) / beta / 2 * id(ℂ^chi)
    psi = cmps(Q, psi.R)
    Q = psi1.Q - log_ovlp(psi1, psi1, beta) / beta / 2 * id(ℂ^chi1)
    psi1 = cmps(Q, psi1.R)
    
    # fidelity
    fidelity = 2*real(log_ovlp(psi, psi1, beta))

    # entanglement spectra
    K = K_mat(psi, psi)
    W, U = eigh(K)
    expK = U * exp(W*beta/2) * U'
    expK = permute(expK, (2, 3), (1, 4))
    _, SK, _ = tsvd(expK, (4, 1), (2, 3))
    entangle_spect = diag(SK.data) / SK.data[1]

    K1 = K_mat(psi, psi1)
    W1, U1 = eigh(K1)
    expK1 = U1 * exp(W1*beta/2) * U1'
    expK1 = permute(expK1, (2, 3), (1, 4))
    _, SK1, _ = tsvd(expK1, (4, 1), (2, 3))
    entangle_spect1 = diag(SK1.data) / SK1.data[1]

    return fidelity, entangle_spect[1:chi1^2], entangle_spect1
    
end