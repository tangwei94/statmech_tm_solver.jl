@timedtestset "test get_matrices rrule" for ix in 1:10
    ψ_arr = rand(ComplexF64, (8, 8, 8))

    function f(arr::Array{ComplexF64, 3})
        ψ = convert_to_cmps(arr)
        Q, R = get_matrices(ψ)
        return norm(get_matrices(ψ)[1]) + norm(get_matrices(ψ)[2])
    end
    
    @test f'(ψ_arr) ≈ FiniteDifferences.grad(central_fdm(5, 1), f, ψ_arr)[1]
end

@timedtestset "test transf_mat and transf_mat_T" for ix in 1:10
    ψ = cmps(rand, 8, 12)
    ϕ = cmps(rand, 9, 12)
    lop = transf_mat(ψ, ϕ)
    lop_T = transf_mat_T(ψ, ϕ)
    v = TensorMap(rand, ComplexF64, ℂ^9, ℂ^8)
    
    Iψ = id(ℂ^8)
    Iϕ = id(ℂ^9)
    
    Qψ_arr, Rψ_arr, Iψ_arr = toarray(ψ.Q), toarray(ψ.R), toarray(Iψ)
    Qϕ_arr, Rϕ_arr, Iϕ_arr = toarray(ϕ.Q), toarray(ϕ.R), toarray(Iϕ)
    v_arr = toarray(v)
    
    lop_arr = ein"db,ca->dcba"(Qϕ_arr, conj(Iψ_arr)) + 
              ein"db,ca->dcba"(Iϕ_arr, conj(Qψ_arr)) +
              ein"dmb,cma->dcba"(Rϕ_arr, conj(Rψ_arr))
    lop_T_arr = ein"dcba->badc"(conj(lop_arr))
    
    lop_v_arr = ein"dcba,ba->dc"(lop_arr, v_arr)
    @test lop_v_arr ≈ toarray(lop(v))
    lop_T_v_arr = ein"badc,dc->ba"(lop_T_arr, v_arr)
    @test lop_T_v_arr ≈ toarray(lop_T(v))
end

# todo: improve the test cases for cmps canonical
@timedtestset "test left canonical for cmps" for ix in 1:10
    ψ = cmps(rand, 8, 3)
    _, ψL = left_canonical(ψ)

    @tensor result[-1; -2] := ψL.Q'[-1, -2] + ψL.Q[-1, -2] + ψL.R[1, 2, -2] * ψL.R'[-1, 1, 2]
    @test isapprox(result, TensorMap(zeros, ComplexF64, ℂ^8, ℂ^8), atol=sqrt(eps()))
end

# todo: improve the test cases for cmps canonical
@timedtestset "test right canonical for cmps" for ix in 1:10
    ψ = cmps(rand, 8, 3)
    _, ψR = right_canonical(ψ)

    @tensor result[-1; -2] := ψR.Q'[-1, -2] + ψR.Q[-1, -2] + ψR.R[-1, 2, 1] * ψR.R'[1, -2, 2]
    @test isapprox(result, TensorMap(zeros, ComplexF64, ℂ^8, ℂ^8), atol=sqrt(eps()))
end

#@timedtestset "normalized after canonicalization" for ix in 1:10
#    ψ = cmps(rand, 9, 4)
#    _, ψR = right_canonical(ψ)
#    X, _ = left_canonical(ψR)
#    _, s, _ = tsvd(X)
#    s = diag(s.data)
#
#    @test isapprox(sum(s .^ 2), 1, rtol=eps()^(1/4))
#
#    ψ = cmps(rand, 9, 4)
#    _, ψL = left_canonical(ψ)
#    Y, _ = right_canonical(ψL)
#    _, s, _ = tsvd(Y)
#    s = diag(s.data)
#
#    @test isapprox(sum(s .^ 2), 1, rtol=eps()^(1/4))
#end
#

@timedtestset "test ovlp and act" for ix in 1:10
    ψ = cmps(rand, 9, 3)
    ϕ = cmps(rand, 5, 3)
    op = cmpo_xxz(0.321)
    idψ = id(ℂ^9)
    idϕ = id(ℂ^5)
    idop = id(ℂ^2)

    Qψ_arr, Qϕ_arr = convert_to_array(ψ.Q), convert_to_array(ϕ.Q)
    Rψ_arr, Rϕ_arr = convert_to_array(ψ.R), convert_to_array(ϕ.R)
    Qop_arr, Lop_arr, Rop_arr = convert_to_array(op.Q), convert_to_array(op.L), convert_to_array(op.R)
    idψ_arr, idϕ_arr, idop_arr = convert_to_array(idψ), convert_to_array(idϕ), convert_to_array(idop)

    T = kron(conj(Qψ_arr), idϕ_arr) + 
        kron(conj(idψ_arr), Qϕ_arr)
    for ix in 1:3
        T += kron(conj(Rψ_arr[:, ix, :]), Rϕ_arr[:, ix, :]) 
    end

    w, _ = eigen(T)
    @test logsumexp(w*6.123) ≈ log_ovlp(ψ, ϕ, 6.123)

    T2 = kron(conj(Qψ_arr), idop_arr, idϕ_arr) +
         kron(conj(idψ_arr), Qop_arr, idϕ_arr) +
         kron(conj(idψ_arr), idop_arr, Qϕ_arr)
    for ix in 1:3
        T2 += kron(conj(Rψ_arr[:, ix, :]), Rop_arr[:, ix, :], idϕ_arr) +
              kron(conj(idψ_arr), (Lop_arr[:, ix, :])', Rϕ_arr[:, ix, :])  
    end
    w2, _ = eigen(T2)
    @test logsumexp(w2*6.123) ≈ log_ovlp(ψ, act(op, ϕ), 6.123)
end

# test rrule for log_ovlp
@timedtestset "rrule for cmps log_ovlp" for ix in 1:10
    ψ_arr = rand(ComplexF64, (3, 4, 3))
    ϕ_arr = rand(ComplexF64, (4, 4, 4))

    function f1(arr::Array{<:Complex})
        ψ = convert_to_cmps(arr)
        return real(log_ovlp(ψ, ψ, 6))
    end
    @test f1'(ψ_arr) ≈ FiniteDifferences.grad(central_fdm(5, 1), f1, ψ_arr)[1]

    function f2(arr::Array{<:Complex})
        ψ = convert_to_cmps(arr)
        ϕ = convert_to_cmps(ϕ_arr)
        return real(log_ovlp(ψ, ϕ, 6))
    end
    @test f2'(ψ_arr) ≈ FiniteDifferences.grad(central_fdm(5, 1), f2, ψ_arr)[1]
end

# test rrule for acting cmpo on cmps
@timedtestset "rrule for acting cmpo on cmps" for ix in 1:10
    ψ_arr = rand(ComplexF64, (3, 4, 3))
    ϕ_arr = rand(ComplexF64, (4, 4, 4))

    op = cmpo_xxz(0.321)

    function f1(arr::Array{<:Complex})
        ψ = convert_to_cmps(arr)
        return real(log_ovlp(ψ, act(op, ψ), 1.234)) 
    end
    @test f1'(ψ_arr) ≈ FiniteDifferences.grad(central_fdm(5, 1), f1, ψ_arr)[1]

    function f2(arr::Array{<:Complex})
        ψ = convert_to_cmps(arr)
        ϕ = convert_to_cmps(ϕ_arr)

        return real(log_ovlp(ψ, act(op, ϕ), 4.321))
    end
    @test f2'(ψ_arr) ≈ FiniteDifferences.grad(central_fdm(5, 1), f2, ψ_arr)[1]

end

# test finite_env and its rrule
@timedtestset "test finite_env and its rrule" for ix in 1:10
    ψ = cmps(rand, 6, 6)
    K = K_mat(ψ, ψ)
    L = rand()
    @test finite_env(K, L) ≈ exp(L*K) / tr(exp(L*K))

    # test rrule
    ψ_arr = rand(ComplexF64, (3, 4, 3))
    ϕ_arr = rand(ComplexF64, (4, 4, 4))
    O = TensorMap(rand, ComplexF64, (ℂ^3)'*ℂ^4, (ℂ^3)'*ℂ^4)

    function f1(arr::Array{<:Complex})
        ψ = convert_to_cmps(arr)
        ϕ = convert_to_cmps(ϕ_arr)
        K = K_mat(ψ, ϕ)
        return tr(finite_env(K, L) * O) |> real
    end

    function f2(arr::Array{<:Complex})
        ψ = convert_to_cmps(arr)
        ϕ = convert_to_cmps(ϕ_arr)
        K = K_mat(ψ, ϕ)
        return norm(finite_env(K, L))
    end
    
    function f3(arr::Array{<:Complex})
        ψ = convert_to_cmps(arr)
        K = K_mat(ψ, ψ)
        return tr(finite_env(K, L) * K) |> real
    end

    @test f1'(ψ_arr) ≈ FiniteDifferences.grad(central_fdm(5, 1), f1, ψ_arr)[1]
    @test f2'(ψ_arr) ≈ FiniteDifferences.grad(central_fdm(5, 1), f2, ψ_arr)[1]
    @test f3'(ψ_arr) ≈ FiniteDifferences.grad(central_fdm(5, 1), f3, ψ_arr)[1]
end

# test normalization
@timedtestset "finite periodic cmps normalization" for ix in 1:10
    ψ = cmps(rand, 8, 8)
    ψ1 = normalize(ψ, 12)

    @test log_ovlp(ψ1, ψ1, 12; sym=true) < 1e-12
end

# test gauge_fixing_proj
@timedtestset "test gauge fixing" for ix in 1:10
    L = 10*rand()
    χ, d = 4, 1
    ψ = cmps(rand, χ, d)
    K = K_mat(ψ, ψ)
    ρ = finite_env(K, L)
    P = gauge_fixing_proj(ψ, L)

    A = cmps(id(ℂ^get_chi(ψ)), copy(ψ.R))
    A = convert_to_tensormap(A)
    V = TensorMap(rand, ComplexF64, ℂ^(d*χ), ℂ^χ)
    G = P * V

    @tensor M[-1, -2; -3, -4] := A'[-3, -1, 1] * G[-2, 1, -4]
    ρM = ρ * M
    @test abs(tr(ρM)) < 1e-14

    @tensor empty[-1; -2] := ρM[1, -1; 1, -2]
    @test findmax(abs.(empty.data))[1] < 1e-14
end 

# test tangent_map 
@timedtestset "test tangent_map" for ix in 1:10
    L = 50 
    χ, d = 4, 1
    p = rand() * 2 * pi / L 
    ψ = 0.01*cmps(rand, χ, d)
    ψ = normalize(ψ, L; sym=false)
    K = K_mat(ψ, ψ)
    W, UR = eig(K)
    UL = inv(UR)

    P = gauge_fixing_proj(ψ, L)

    A = cmps(id(ℂ^get_chi(ψ)), copy(ψ.R))
    A = convert_to_tensormap(A)
    V = TensorMap(rand, ComplexF64, ℂ^(d*χ), ℂ^χ)
    
    lop = tangent_map(ψ, L, p)

    #manual integration
    @tensor A1[-1, -2; -3, -4] := V[1, -4] * P[-2, 2, 1] * A'[-3, -1, 2]
    normψ = tr(exp(L*K))

    function fVc2(tau::Real)
        A2 = permute(exp(K*(L-tau)) * A1 * exp(K*(tau)), (2, 3), (4, 1))
        @tensor Vtmp[-1; -2] := A2[1, 3, 2, -2] * A[2, 4, 1] * P'[-1, 3, 4]
        return exp(-p*tau*im) * Vtmp 
    end
    Vc2 = quadgk(fVc2, 0, L)[1]

    ρ = permute(exp(L*K), (2, 3), (4, 1))
    D = id(ℂ^(d+1))
    D.data[1] = 0
    @tensor Vd2[-1; -2] := ρ[3, 5, 2, -2] * P[2, 4, 1] * V[1, 3] * D[6, 4] * P'[-1, 5, 6]
    V2 = L*(Vc2 + Vd2) / normψ
    
    # compare
    @test norm(lop(V) - V2) / L < 1e-14
end

@timedtestset "test util functions for h tangent map" for ix in 1:10
    L = rand()
    p, q = 2*pi/L, 4*pi/L
    δ=1e-6
    theta2 = (a, b) -> statmech_tm_solver.theta2(L, a, b)
    theta3 = (a, b, c) -> statmech_tm_solver.theta3(L, a, b, c)

    M = TensorMap(rand, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
    A = TensorMap(rand, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
    B = TensorMap(rand, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
    C = TensorMap(rand, ComplexF64, ℂ^2*ℂ^2, ℂ^2*ℂ^2)
    W, VR = eig(M)
    VL = inv(VR)
    Wvec = diag(W.data)
    At = VL * A * VR
    Bt = VL * B * VR
    Ct = VL * C * VR

    # test elem_mult_f1 and theta2
    @test isapprox(theta2(1, 2), theta2(2, 1))
    @test isapprox(theta2(1, 1+δ), theta2(1, 1); atol=sqrt(δ))

    At1 = statmech_tm_solver.elem_mult_f1(At, (ix, iy) -> theta2(Wvec[ix], Wvec[iy] + im*p))    
    @test tr(At1 * Bt) ≈ quadgk(x1 -> exp(im*p*x1)tr(A * exp(x1*M) * B * exp((L-x1)*M)), 0, L)[1]

    # test elem_mult_f2 and theta3
    @test theta3(1, 2, 3) ≈ theta3(2, 3, 1) ≈ theta3(3, 1, 2)
    @test theta3(1, 1, 2) ≈ theta3(2, 1, 1) ≈ theta3(1, 2, 1)

    @test isapprox(theta3(1, 1+δ, 2), theta3(1+δ, 1, 2); atol=δ)
    @test isapprox(theta3(1, 1, 2), theta3(1+δ, 1, 2); atol=sqrt(δ))
    @test isapprox(theta3(1, 1, 1), theta3(1, 1+δ, 1); atol=sqrt(δ))
    @test isapprox(theta3(1, 1-δ, 1+δ), theta3(1, 1, 1); atol=sqrt(δ))

    ABt = statmech_tm_solver.elem_mult_f2(At, Bt, (ix, iz, iy) -> theta3(Wvec[ix] - im*p, Wvec[iz] - im*(p+q), Wvec[iy]))
    @test tr(ABt * Ct) ≈ quadgk(x1 -> quadgk(
        x2 -> exp(im*p*x1) * exp(-im*(p+q)*x2) * tr(A * exp(x2*M) * B * exp((x1 - x2)*M) * C * exp((L-x1)*M) ), 
        0, x1)[1], 0, L)[1] 
end

@timedtestset "test lieb liniger h tangent map; test for d=1" for ix in 1:10
    
    k0, μ, c = rand(3)
    χ, d, L = 2, 1, rand()
    p, q = 2*pi/L, 2*2*pi/L

    ψ = cmps(rand, χ, d)
    ψ = normalize(ψ, L; sym=false)
    P = gauge_fixing_proj(ψ, L)
    Vx2 = TensorMap(rand, ComplexF64, ℂ^(χ*d), ℂ^χ)
    Vx1= TensorMap(rand, ComplexF64, ℂ^(χ*d), ℂ^χ)

    # overlap by tangent map function
    lop = lieb_liniger_h_tangent_map(ψ, p, q, c, L, μ; k0=k0)
    aaa = tr(Vx2' * lop(Vx1))

    # manual integration + alternative construction of My
    ψtn = convert_to_tensormap(ψ)
    K = K_mat(ψ, ψ)
    
    # to be used in construction of M matrices (My, Mx1, Mx2, see below) with tensorkit
    A = cmps(id(ℂ^χ), copy(ψ.R))
    A = convert_to_tensormap(A)
    DR = id(ℂ^(d+1))
    DR.data[1] = 0
    DQ = zero(DR)
    DQ.data[1] = 1 
    DQRRQ = zeros(d+1, d+1, d+1, d+1)
    DRQQR = zeros(d+1, d+1, d+1, d+1)
    for ix in 2:d+1
        DQRRQ[1, ix, ix, 1] = 1
        DRQQR[ix, 1, 1, ix] = 1
    end
    DQRRQ = TensorMap(DQRRQ, ℂ^(d+1)*ℂ^(d+1), ℂ^(d+1)*ℂ^(d+1))
    DRQQR = TensorMap(DRQQR, ℂ^(d+1)*ℂ^(d+1), ℂ^(d+1)*ℂ^(d+1))

    @tensor QR_comm[-1, -2; -3, -4] := DQ[-1, -3] * DR[-2, -4] + DR[-1, -3] * DQ[-2, -4] -
                                       DQRRQ[-1, -2, -3, -4] - DRQQR[-1, -2, -3, -4]
    @tensor QR_comm2[-1, -2; -3, -4] := DQ[-1, -3] * DR[-2, -4] - DRQQR[-1, -2, -3, -4]
    @tensor QR_comm3[-1, -2; -3, -4] := DQ[-1, -3] * DR[-2, -4] - DQRRQ[-1, -2, -3, -4]

    # to be used in construction of M matrices by using Q, R matrices, assuming d = 2
    Q = ψ.Q
    Iψ = id(ℂ^χ)
    R = TensorMap(convert_to_array(ψ.R)[:, 1, :], ℂ^χ, ℂ^χ)
    tangent1, tangent2 = P * Vx1, P * Vx2
    V1  = TensorMap(convert_to_array(tangent1)[:, 1, :], ℂ^χ, ℂ^χ)
    W1  = TensorMap(convert_to_array(tangent1)[:, 2, :], ℂ^χ, ℂ^χ)
    V2 = TensorMap(convert_to_array(tangent2)[:, 1, :], ℂ^χ, ℂ^χ)
    W2 = TensorMap(convert_to_array(tangent2)[:, 2, :], ℂ^χ, ℂ^χ)
    tnp = (A1, A2) -> (@tensor Aout[-1, -2; -3, -4] := A1'[-3, -1] * A2[-2, -4])
    comm = (A1, A2) -> (A1 * A2 - A2 * A1)

    # Mx's
    @tensor Mx1[-1, -2; -3, -4] := A'[-3, -1, 2] * P[-2, 2, 1] * Vx1[1, -4]
    @tensor Mx2[-1, -2; -3, -4] := P'[1, -1, 2] * Vx2'[-3, 1] * A[-2, 2, -4]
    @tensor Mxx[-1, -2; -3, -4] := P'[1, -1, 3] * Vx2'[-3, 1] * DR[3, 4] * P[-2, 4, 2] * Vx1[2, -4] 
    @test Mx1 ≈ tnp(Iψ, V1) + tnp(R, W1) 
    @test Mx2 ≈ tnp(V2, Iψ) + tnp(W2, R)
    @test Mxx ≈ tnp(W2, W1)

    # x != x' != y
    @tensor My[-1, -2; -3, -4] := -μ * ψtn'[-3, -1, 1] * DR[1, 2] * ψtn[-2, 2, -4] +
                                  k0 * ψtn'[1, -1, 3] * ψtn'[-3, 1, 4] * QR_comm[3, 4, 5, 6] * ψtn[-2, 5, 2] * ψtn[2, 6, -4] + 
                                  c * ψtn'[1, -1, 3] * ψtn'[-3, 1, 4] * DR[3, 5] * DR[4, 6] * ψtn[-2, 5, 2] * ψtn[2, 6, -4]
    @test My ≈ -μ * tnp(R, R) +
               k0 * tnp(comm(Q, R), comm(Q, R)) + 
               c * tnp(R*R, R*R) 
    res1 = 
    quadgk(x1 -> quadgk(
        x2 -> exp(-im*(p+q)*x2) * exp(im*p*x1) * tr(My * exp(x2*K) * Mx2 * exp((x1 - x2)*K) * Mx1 * exp((L-x1)*K) ), 
        0, x1)[1], 0, L)[1] +
    quadgk(x1 -> quadgk(
        x2 -> exp(-im*(p+q)*x2) * exp(im*p*x1) * tr(My * exp(x1*K) * Mx1 * exp((x2 - x1)*K) * Mx2 * exp((L-x2)*K) ),
        x1, L)[1], 0, L)[1]

    # y != x == x' 
    res1_a = quadgk(x1 -> exp(-im*q*x1) * tr(My * exp(x1*K) * Mxx * exp((L-x1)*K)), 0, L)[1]

    # y == x != x'
    @tensor My[-1, -2; -3, -4] := -μ * ψtn'[-3, -1, 2] * DR[2, 3] * P[-2, 3, 1] * Vx1[1, -4] +
                                  k0 * ψtn'[3, -1, 4] * ψtn'[-3, 3, 5] * QR_comm[4, 5, 6, 7] * P[-2, 6, 1] * Vx1[1, 2] * ψtn[2, 7, -4] +
                                  k0 * ψtn'[3, -1, 4] * ψtn'[-3, 3, 5] * QR_comm[4, 5, 6, 7] * ψtn[-2, 6, 2] * P[2, 7, 1] * Vx1[1, -4] +
                                  k0 * ψtn'[3, -1, 4] * ψtn'[-3, 3, 5] * DQ[4, 6] * DR[5, 7] * (im*p)*A[-2, 6, 2] * P[2, 7, 1] * Vx1[1, -4] - 
                                  k0 * ψtn'[3, -1, 4] * ψtn'[-3, 3, 5] * DRQQR[4, 5, 6, 7] * (im*p)*A[-2, 6, 2] * P[2, 7, 1] * Vx1[1, -4] +
                                  c * ψtn'[1, -1, 4] * ψtn'[-3, 1, 5] * DR[4, 6] * DR[5, 7] * ψtn[-2, 6, 3] * P[3, 7, 2] * Vx1[2, -4] + 
                                  c * ψtn'[1, -1, 4] * ψtn'[-3, 1, 5] * DR[4, 6] * DR[5, 7] * P[-2, 6, 2] * Vx1[2, 3] * ψtn[3, 7, -4]
    @test My ≈ -μ * tnp(R, W1) + 
               k0 * tnp(comm(Q, R), comm(V1, R) + comm(Q, W1) + im*p*W1) + 
               c * tnp(R*R, R*W1 + W1*R) 
    res2 = quadgk(x2 -> exp(-im*(p+q)*x2) * tr(My * exp(x2*K) * Mx2 * exp((L-x2)*K)), 0, L)[1]

    # y == x' != x
    @tensor My[-1, -2; -3, -4] := -μ * P'[1, -1, 2] * Vx2'[-3, 1] * DR[2, 3] * ψtn[-2, 3, -4] + 
                                  k0 * P'[1, -1, 4] * Vx2'[2, 1] * ψtn'[-3, 2, 5] * QR_comm[4, 5, 6, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4] + 
                                  k0 * ψtn'[2, -1, 4] * P'[1, 2, 5] * Vx2'[-3, 1] * QR_comm[4, 5, 6, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4] + 
                                  k0 * (-im*(p+q))*A'[2, -1, 4] * P'[1, 2, 5] * Vx2'[-3, 1] * DQ[4, 6] * DR[5, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4] -
                                  k0 * (-im*(p+q))*A'[2, -1, 4] * P'[1, 2, 5] * Vx2'[-3, 1] * DQRRQ[4, 5, 6, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4] +
                                  c * ψtn'[2, -1, 4] * P'[1, 2, 5] * Vx2'[-3, 1] * DR[4, 6] * DR[5, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4] + 
                                  c * P'[1, -1, 4] * Vx2'[2, 1] * ψtn'[-3, 2, 5] * DR[4, 6] * DR[5, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4]
    @test My ≈ -μ * tnp(W2, R) +
               k0 * tnp(comm(V2, R) + comm(Q, W2) + im*(p+q)*W2, comm(Q, R)) +
               c * tnp(R*W2 + W2*R, R*R) 
    res3 = quadgk(x1 -> exp(im*p*x1) * tr(My * exp(x1*K) * Mx1 * exp((L-x1)*K)), 0, L)[1]

    # y == x' == x
    @tensor My[-1, -2; -3, -4] := -μ * P'[1, -1, 3] * Vx2'[-3, 1] * DR[3, 4] * P[-2, 4, 2] * Vx1[2, -4] +
                                  k0 * P'[1, -1, 5] * Vx2'[2, 1] * ψtn'[-3, 2, 6] * QR_comm[5, 6, 7, 8] * P[-2, 7, 3] * Vx1[3, 4] * ψtn[4, 8, -4] +
                                  k0 * P'[1, -1, 5] * Vx2'[2, 1] * ψtn'[-3, 2, 6] * QR_comm[5, 6, 7, 8] * ψtn[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] +
                                  k0 * ψtn'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * QR_comm[5, 6, 7, 8] * ψtn[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] +
                                  k0 * ψtn'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * QR_comm[5, 6, 7, 8] * P[-2, 7, 3] * Vx1[3, 4] * ψtn[4, 8, -4] +
                                  k0 * P'[1, -1, 5] * Vx2'[2, 1] * ψtn'[-3, 2, 6] * QR_comm2[5, 6, 7, 8] * (im*p)*A[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] + 
                                  k0 * ψtn'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * QR_comm2[5, 6, 7, 8] * (im*p)*A[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] +
                                  k0 * (-im*(p+q))*A'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * QR_comm3[5, 6, 7, 8] * ψtn[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] + 
                                  k0 * (-im*(p+q))*A'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * QR_comm3[5, 6, 7, 8] * P[-2, 7, 3] * Vx1[3, 4] * ψtn[4, 8, -4] + 
                                  k0 * (-im*(p+q))*A'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * DQ[5, 7] * DR[6, 8] * (im*p)*A[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] +
                                  c * ψtn'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * DR[5, 7] * DR[6, 8] * ψtn[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] + 
                                  c * P'[1, -1, 5] * Vx2'[2, 1] * ψtn'[-3, 2, 6] * DR[5, 7] * DR[6, 8] * ψtn[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] + 
                                  c * ψtn'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * DR[5, 7] * DR[6, 8] * P[-2, 7, 3] * Vx1[3, 4] * ψtn[4, 8, -4] + 
                                  c * P'[1, -1, 5] * Vx2'[2, 1] * ψtn'[-3, 2, 6] * DR[5, 7] * DR[6, 8] * P[-2, 7, 3] * Vx1[3, 4] * ψtn[4, 8, -4]
    @test My ≈ -μ * tnp(W2, W1) +
               k0 * tnp(comm(V2, R) + comm(Q, W2) + im*(p+q)*W2, comm(V1, R) + comm(Q, W1) + im*p*W1) + 
               c * tnp(R*W2 + W2*R, R*W1 + W1*R)  
    res4 = tr(exp(L*K) * My)

    bbb = L*(res1 + res1_a + res2 + res3 + res4) 

    @test aaa ≈ bbb
end

@timedtestset "test lieb liniger h tangent map; test for d=2" for ix in 1:10
    
    k0, μ, c = rand(3)
    χ, d, L = 2, 2, rand()
    p, q = 2*pi/L, 2*2*pi/L

    ψ = cmps(rand, χ, d)
    ψ = normalize(ψ, L; sym=false)
    P = gauge_fixing_proj(ψ, L)
    Vx2 = TensorMap(rand, ComplexF64, ℂ^(χ*d), ℂ^χ)
    Vx1= TensorMap(rand, ComplexF64, ℂ^(χ*d), ℂ^χ)

    # overlap by tangent map function
    lop = lieb_liniger_h_tangent_map(ψ, p, q, c, L, μ; k0=k0)
    aaa = tr(Vx2' * lop(Vx1))

    # manual integration + alternative construction of My
    ψtn = convert_to_tensormap(ψ)
    K = K_mat(ψ, ψ)
    
    # to be used in construction of M matrices (My, Mx1, Mx2, see below) with tensorkit
    A = cmps(id(ℂ^χ), copy(ψ.R))
    A = convert_to_tensormap(A)
    DR = id(ℂ^(d+1))
    DR.data[1] = 0
    DQ = zero(DR)
    DQ.data[1] = 1 
    DQRRQ = zeros(d+1, d+1, d+1, d+1)
    DRQQR = zeros(d+1, d+1, d+1, d+1)
    for ix in 2:d+1
        DQRRQ[1, ix, ix, 1] = 1
        DRQQR[ix, 1, 1, ix] = 1
    end
    DQRRQ = TensorMap(DQRRQ, ℂ^(d+1)*ℂ^(d+1), ℂ^(d+1)*ℂ^(d+1))
    DRQQR = TensorMap(DRQQR, ℂ^(d+1)*ℂ^(d+1), ℂ^(d+1)*ℂ^(d+1))

    @tensor QR_comm[-1, -2; -3, -4] := DQ[-1, -3] * DR[-2, -4] + DR[-1, -3] * DQ[-2, -4] -
                                       DQRRQ[-1, -2, -3, -4] - DRQQR[-1, -2, -3, -4]
    @tensor QR_comm2[-1, -2; -3, -4] := DQ[-1, -3] * DR[-2, -4] - DRQQR[-1, -2, -3, -4]
    @tensor QR_comm3[-1, -2; -3, -4] := DQ[-1, -3] * DR[-2, -4] - DQRRQ[-1, -2, -3, -4]

    # to be used in construction of M matrices by using Q, R matrices, assuming d = 2
    Q = ψ.Q
    Iψ = id(ℂ^χ)
    Ra = TensorMap(convert_to_array(ψ.R)[:, 1, :], ℂ^χ, ℂ^χ)
    Rb = TensorMap(convert_to_array(ψ.R)[:, 2, :], ℂ^χ, ℂ^χ)
    tangent1, tangent2 = P * Vx1, P * Vx2
    V1  = TensorMap(convert_to_array(tangent1)[:, 1, :], ℂ^χ, ℂ^χ)
    Wa1  = TensorMap(convert_to_array(tangent1)[:, 2, :], ℂ^χ, ℂ^χ)
    Wb1  = TensorMap(convert_to_array(tangent1)[:, 3, :], ℂ^χ, ℂ^χ)
    V2 = TensorMap(convert_to_array(tangent2)[:, 1, :], ℂ^χ, ℂ^χ)
    Wa2 = TensorMap(convert_to_array(tangent2)[:, 2, :], ℂ^χ, ℂ^χ)
    Wb2 = TensorMap(convert_to_array(tangent2)[:, 3, :], ℂ^χ, ℂ^χ)
    tnp = (A1, A2) -> (@tensor Aout[-1, -2; -3, -4] := A1'[-3, -1] * A2[-2, -4])
    comm = (A1, A2) -> (A1 * A2 - A2 * A1)

    # Mx's
    @tensor Mx1[-1, -2; -3, -4] := A'[-3, -1, 2] * P[-2, 2, 1] * Vx1[1, -4]
    @tensor Mx2[-1, -2; -3, -4] := P'[1, -1, 2] * Vx2'[-3, 1] * A[-2, 2, -4]
    @tensor Mxx[-1, -2; -3, -4] := P'[1, -1, 3] * Vx2'[-3, 1] * DR[3, 4] * P[-2, 4, 2] * Vx1[2, -4] 
    @test Mx1 ≈ tnp(Iψ, V1) + tnp(Ra, Wa1) + tnp(Rb, Wb1)
    @test Mx2 ≈ tnp(V2, Iψ) + tnp(Wa2, Ra) + tnp(Wb2, Rb)
    @test Mxx ≈ tnp(Wa2, Wa1) + tnp(Wb2, Wb1)

    # x != x' != y
    @tensor My[-1, -2; -3, -4] := -μ * ψtn'[-3, -1, 1] * DR[1, 2] * ψtn[-2, 2, -4] +
                                  k0 * ψtn'[1, -1, 3] * ψtn'[-3, 1, 4] * QR_comm[3, 4, 5, 6] * ψtn[-2, 5, 2] * ψtn[2, 6, -4] + 
                                  c * ψtn'[1, -1, 3] * ψtn'[-3, 1, 4] * DR[3, 5] * DR[4, 6] * ψtn[-2, 5, 2] * ψtn[2, 6, -4]
    @test My ≈ -μ * (tnp(Ra, Ra) + tnp(Rb, Rb)) +
               k0 * (tnp(comm(Q, Ra), comm(Q, Ra)) + tnp(comm(Q, Rb), comm(Q, Rb))) + 
               c * (tnp(Ra*Ra, Ra*Ra) + tnp(Rb*Rb, Rb*Rb) + tnp(Ra*Rb, Ra*Rb) + tnp(Rb*Ra, Rb*Ra)) 
    res1 = 
    quadgk(x1 -> quadgk(
        x2 -> exp(-im*(p+q)*x2) * exp(im*p*x1) * tr(My * exp(x2*K) * Mx2 * exp((x1 - x2)*K) * Mx1 * exp((L-x1)*K) ), 
        0, x1)[1], 0, L)[1] +
    quadgk(x1 -> quadgk(
        x2 -> exp(-im*(p+q)*x2) * exp(im*p*x1) * tr(My * exp(x1*K) * Mx1 * exp((x2 - x1)*K) * Mx2 * exp((L-x2)*K) ),
        x1, L)[1], 0, L)[1]


    # y != x == x' 
    res1_a = quadgk(x1 -> exp(-im*q*x1) * tr(My * exp(x1*K) * Mxx * exp((L-x1)*K)), 0, L)[1]

    # y == x != x'
    @tensor My[-1, -2; -3, -4] := -μ * ψtn'[-3, -1, 2] * DR[2, 3] * P[-2, 3, 1] * Vx1[1, -4] +
                                  k0 * ψtn'[3, -1, 4] * ψtn'[-3, 3, 5] * QR_comm[4, 5, 6, 7] * P[-2, 6, 1] * Vx1[1, 2] * ψtn[2, 7, -4] +
                                  k0 * ψtn'[3, -1, 4] * ψtn'[-3, 3, 5] * QR_comm[4, 5, 6, 7] * ψtn[-2, 6, 2] * P[2, 7, 1] * Vx1[1, -4] +
                                  k0 * ψtn'[3, -1, 4] * ψtn'[-3, 3, 5] * DQ[4, 6] * DR[5, 7] * (im*p)*A[-2, 6, 2] * P[2, 7, 1] * Vx1[1, -4] - 
                                  k0 * ψtn'[3, -1, 4] * ψtn'[-3, 3, 5] * DRQQR[4, 5, 6, 7] * (im*p)*A[-2, 6, 2] * P[2, 7, 1] * Vx1[1, -4] +
                                  c * ψtn'[1, -1, 4] * ψtn'[-3, 1, 5] * DR[4, 6] * DR[5, 7] * ψtn[-2, 6, 3] * P[3, 7, 2] * Vx1[2, -4] + 
                                  c * ψtn'[1, -1, 4] * ψtn'[-3, 1, 5] * DR[4, 6] * DR[5, 7] * P[-2, 6, 2] * Vx1[2, 3] * ψtn[3, 7, -4]
    @test My ≈ -μ * (tnp(Ra, Wa1) + tnp(Rb, Wb1)) + 
               k0 * (tnp(comm(Q, Ra), comm(V1, Ra) + comm(Q, Wa1) + im*p*Wa1) + tnp(comm(Q, Rb), comm(V1, Rb) + comm(Q, Wb1) + im*p*Wb1)) + 
               c * (tnp(Ra*Ra, Ra*Wa1 + Wa1*Ra) + tnp(Rb*Rb, Rb*Wb1 + Wb1*Rb) + tnp(Ra*Rb, Ra*Wb1 + Wa1*Rb) + tnp(Rb*Ra, Rb*Wa1 + Wb1*Ra))
    res2 = quadgk(x2 -> exp(-im*(p+q)*x2) * tr(My * exp(x2*K) * Mx2 * exp((L-x2)*K)), 0, L)[1]

    # y == x' != x
    @tensor My[-1, -2; -3, -4] := -μ * P'[1, -1, 2] * Vx2'[-3, 1] * DR[2, 3] * ψtn[-2, 3, -4] + 
                                  k0 * P'[1, -1, 4] * Vx2'[2, 1] * ψtn'[-3, 2, 5] * QR_comm[4, 5, 6, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4] + 
                                  k0 * ψtn'[2, -1, 4] * P'[1, 2, 5] * Vx2'[-3, 1] * QR_comm[4, 5, 6, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4] + 
                                  k0 * (-im*(p+q))*A'[2, -1, 4] * P'[1, 2, 5] * Vx2'[-3, 1] * DQ[4, 6] * DR[5, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4] -
                                  k0 * (-im*(p+q))*A'[2, -1, 4] * P'[1, 2, 5] * Vx2'[-3, 1] * DQRRQ[4, 5, 6, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4] +
                                  c * ψtn'[2, -1, 4] * P'[1, 2, 5] * Vx2'[-3, 1] * DR[4, 6] * DR[5, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4] + 
                                  c * P'[1, -1, 4] * Vx2'[2, 1] * ψtn'[-3, 2, 5] * DR[4, 6] * DR[5, 7] * ψtn[-2, 6, 3] * ψtn[3, 7, -4]
    @test My ≈ -μ * (tnp(Wa2, Ra) + tnp(Wb2, Rb)) +
               k0 * (tnp(comm(V2, Ra) + comm(Q, Wa2) + im*(p+q)*Wa2, comm(Q, Ra)) + tnp(comm(V2, Rb) + comm(Q, Wb2) + im*(p+q)*Wb2, comm(Q, Rb))) +
               c * (tnp(Ra*Wa2 + Wa2*Ra, Ra*Ra) + tnp(Rb*Wb2 + Wb2*Rb, Rb*Rb) + tnp(Ra*Wb2 + Wa2*Rb, Ra*Rb) + tnp(Rb*Wa2 + Wb2*Ra, Rb*Ra))
    res3 = quadgk(x1 -> exp(im*p*x1) * tr(My * exp(x1*K) * Mx1 * exp((L-x1)*K)), 0, L)[1]

    # y == x' == x
    @tensor My[-1, -2; -3, -4] := -μ * P'[1, -1, 3] * Vx2'[-3, 1] * DR[3, 4] * P[-2, 4, 2] * Vx1[2, -4] +
                                  k0 * P'[1, -1, 5] * Vx2'[2, 1] * ψtn'[-3, 2, 6] * QR_comm[5, 6, 7, 8] * P[-2, 7, 3] * Vx1[3, 4] * ψtn[4, 8, -4] +
                                  k0 * P'[1, -1, 5] * Vx2'[2, 1] * ψtn'[-3, 2, 6] * QR_comm[5, 6, 7, 8] * ψtn[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] +
                                  k0 * ψtn'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * QR_comm[5, 6, 7, 8] * ψtn[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] +
                                  k0 * ψtn'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * QR_comm[5, 6, 7, 8] * P[-2, 7, 3] * Vx1[3, 4] * ψtn[4, 8, -4] +
                                  k0 * P'[1, -1, 5] * Vx2'[2, 1] * ψtn'[-3, 2, 6] * QR_comm2[5, 6, 7, 8] * (im*p)*A[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] + 
                                  k0 * ψtn'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * QR_comm2[5, 6, 7, 8] * (im*p)*A[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] +
                                  k0 * (-im*(p+q))*A'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * QR_comm3[5, 6, 7, 8] * ψtn[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] + 
                                  k0 * (-im*(p+q))*A'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * QR_comm3[5, 6, 7, 8] * P[-2, 7, 3] * Vx1[3, 4] * ψtn[4, 8, -4] + 
                                  k0 * (-im*(p+q))*A'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * DQ[5, 7] * DR[6, 8] * (im*p)*A[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] +
                                  c * ψtn'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * DR[5, 7] * DR[6, 8] * ψtn[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] + 
                                  c * P'[1, -1, 5] * Vx2'[2, 1] * ψtn'[-3, 2, 6] * DR[5, 7] * DR[6, 8] * ψtn[-2, 7, 4] * P[4, 8, 3] * Vx1[3, -4] + 
                                  c * ψtn'[2, -1, 5] * P'[1, 2, 6] * Vx2'[-3, 1] * DR[5, 7] * DR[6, 8] * P[-2, 7, 3] * Vx1[3, 4] * ψtn[4, 8, -4] + 
                                  c * P'[1, -1, 5] * Vx2'[2, 1] * ψtn'[-3, 2, 6] * DR[5, 7] * DR[6, 8] * P[-2, 7, 3] * Vx1[3, 4] * ψtn[4, 8, -4]
    @test My ≈ -μ * (tnp(Wa2, Wa1) + tnp(Wb2, Wb1)) +
               k0 * (tnp(comm(V2, Ra) + comm(Q, Wa2) + im*(p+q)*Wa2, comm(V1, Ra) + comm(Q, Wa1) + im*p*Wa1) + 
                     tnp(comm(V2, Rb) + comm(Q, Wb2) + im*(p+q)*Wb2, comm(V1, Rb) + comm(Q, Wb1) + im*p*Wb1)) +
               c * (tnp(Ra*Wa2 + Wa2*Ra, Ra*Wa1 + Wa1*Ra) + 
                    tnp(Rb*Wb2 + Wb2*Rb, Rb*Wb1 + Wb1*Rb) + 
                    tnp(Ra*Wb2 + Wa2*Rb, Ra*Wb1 + Wa1*Rb) +
                    tnp(Rb*Wa2 + Wb2*Ra, Rb*Wa1 + Wb1*Ra))
    res4 = tr(exp(L*K) * My)

    bbb = L*(res1 + res1_a + res2 + res3 + res4) 

    @test aaa ≈ bbb
end
