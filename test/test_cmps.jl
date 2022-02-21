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
    L = rand()
    χ, d = 4, 1
    ψ = cmps(rand, χ, d)
    K = K_mat(ψ, ψ)
    ρ = finite_env(K, L)
    P = gauge_fixing_proj(ψ, L)

    A = cmps(id(ℂ^get_chi(ψ)), copy(ψ.R))
    A = convert_to_tensormap(A)
    V = Tensor(rand, ComplexF64, ℂ^χ*ℂ^(d*χ))

    @tensor M[-1, -2; -3, -4] := A'[-3, -1, 2] * V[-2, 1] * P[2, -4, 1]
    @test abs(tr(M * ρ)) < 1e-14
    @tensor M1[-1; -2] := (M*ρ)[1, -1; 1, -2]
    @test findmax(abs.(M1.data))[1] < 1e-14
end 

# test precond_grad 
@timedtestset "test tangent_map" for ix in 1:10
    L = 50 
    χ, d = 4, 1
    ψ = 0.01*cmps(rand, χ, d)
    #ψ = ψm
    ψ = normalize(ψ, L; sym=false)
    K = K_mat(ψ, ψ)
    W, UR = eig(K)
    UL = inv(UR)

    P = gauge_fixing_proj(ψ, L)

    A = cmps(id(ℂ^get_chi(ψ)), copy(ψ.R))
    A = convert_to_tensormap(A)
    V = TensorMap(rand, ComplexF64, ℂ^(d*χ), ℂ^χ)
    
    lop = tangent_map(ψ, L)
    lop(V)

    #W, _ = eigsolve(lop, V, 12); @show W

    #manual integration
    @tensor A1[-1, -2; -3, -4] := V[1, -4] * P[-2, 2, 1] * A'[-3, -1, 2]
    Vc2 = 0*similar(V)
    normψ = tr(exp(L*K))
    num_div = 10000
    for ix in 1:num_div
        tau = ix * L / num_div
        A2 = permute(exp(K*(L-tau)) * A1 * exp(K*tau), (2, 3), (4, 1))
        @tensor Vtmp[-1; -2] := A2[1, 3, 2, -2] * A[2, 4, 1] * P'[-1, 3, 4] 
        Vc2 = Vc2 + L*Vtmp / num_div / normψ
    end
    ρ = permute(exp(L*K) / normψ, (2, 3), (4, 1))
    D = id(ℂ^(d+1))
    D.data[1] = 0
    @tensor Vd2[-1; -2] := ρ[3, 5, 2, -2] * P[2, 4, 1] * V[1, 3] * D[6, 4] * P'[-1, 5, 6]
    V2 = Vc2 + Vd2

    @test norm(lop(V) - V2) < 1e-5
end