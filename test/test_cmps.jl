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
