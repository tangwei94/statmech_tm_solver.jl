# MPO for the triangular AF Ising
t = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^2, ℂ^2)
p = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2*ℂ^2)
t[1, 1, 2] = 1
t[1, 2, 1] = 1
t[2, 1, 1] = 1
t[2, 2, 1] = 1
t[2, 1, 2] = 1
t[1, 2, 2] = 1
p[1, 1, 1] = 1
p[2, 2, 2] = 1
T = t*p

T_arr = reshape(T.data, (2, 2, 2, 2))

@timedtestset "test act ($ix)" for ix in 1:5

    psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
    psi_arr = reshape(psi.data, (5, 2, 5))
    Tpsi = act(T, psi)
    Tpsi_arr = reshape(ein"acmd,bme->abcde"(T_arr, psi_arr), (20, 10))
    @test Tpsi_arr ≈ Tpsi.data

end

# test rrule of act
@timedtestset "test rrule for act ($ix)" for ix in 1:5
    function f_with_OMEinsum(psi_arr::Array{ComplexF64, 3})
        Tpsi_arr = reshape(ein"acmd,bme->abcde"(T_arr, psi_arr), (10, 2, 10))
        result = ein"abc,abc->"(conj(Tpsi_arr), Tpsi_arr) |> real
        return result[1]
    end
    function f(psi::TensorMap{ComplexSpace, 2, 1})
        Tpsi = act(T, psi)
        return dot(Tpsi, Tpsi) |> real
    end
    
    psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
    psi_arr = reshape(psi.data, (5, 2, 5))
    @test f(psi) ≈ f_with_OMEinsum(psi_arr)
    
    grad_von_OMEinsum = gradient(f_with_OMEinsum, psi_arr)[1]
    grad = reshape(gradient(f, psi)[1].data, (5, 2, 5))
    @test grad_von_OMEinsum ≈ conj.(grad)
end

@timedtestset "test transf_mat constructions ($ix)" for ix in 1:5
    psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
    phi = TensorMap(rand, ComplexF64, ℂ^6*ℂ^2, ℂ^6)
    v = TensorMap(rand, ComplexF64, ℂ^6, ℂ^5)
    lop = transf_mat(psi, phi)
    lop_T = transf_mat_T(psi, phi)
    
    psi_arr = toarray(psi)
    phi_arr = toarray(phi)
    v_arr = toarray(v)
    lop_arr = ein"aec,bed->abcd"(conj.(psi_arr), phi_arr)
    
    lop_v = ein"abcd,dc->ba"(lop_arr, v_arr)
    lopT_v = ein"abcd,ba->dc"(conj.(lop_arr), v_arr)
    
    lop_T(v)
    
    @test lop_v ≈ lop(v) |> toarray
    @test lopT_v ≈ lop_T(v) |> toarray
end

@timedtestset "test ovlp and its rrule" for ix in 1:5
    function ovlp_von_arr(psi_arr::Array{ComplexF64, 3}, phi_arr::Array{ComplexF64, 3})
        lop_arr = ein"aec,bed->abcd"(conj.(psi_arr), phi_arr)
        lop_arr = reshape(lop_arr, (30, 30))
        w_von_arr = eigvals(lop_arr) |> last
        return w_von_arr
    end
    
    psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
    phi = TensorMap(rand, ComplexF64, ℂ^6*ℂ^2, ℂ^6)
    psi_arr = toarray(psi)
    phi_arr = toarray(phi)
    
    @test ovlp(psi, phi) ≈ ovlp_von_arr(psi_arr, phi_arr)
    
    psi_pushback1 = gradient(arr->real(ovlp_von_arr(arr, phi_arr)), psi_arr)[1]
    psi_pushback2 = gradient(arr->real(ovlp(arr_to_TensorMap(arr), phi)), psi_arr)[1]
    @test psi_pushback1 ≈ psi_pushback2
    
    phi_pushback1 = gradient(arr->real(ovlp_von_arr(psi_arr, arr)), phi_arr)[1]
    phi_pushback2 = gradient(arr->real(ovlp(psi, arr_to_TensorMap(arr))), phi_arr)[1]
    @test phi_pushback1 ≈ phi_pushback2

end