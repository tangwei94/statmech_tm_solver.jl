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
    ψ_arr = rand(ComplexF64, (6, 4, 6))
    ϕ_arr = rand(ComplexF64, (6, 4, 6))

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
