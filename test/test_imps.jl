# MPO for the triangular AF Ising
T = mpo_triangular_AF_ising()

T_arr = reshape(T.data, (2, 2, 2, 2))

@timedtestset "test act ($ix)" for ix in 1:10

    psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
    psi_arr = reshape(psi.data, (5, 2, 5))
    Tpsi = act(T, psi)
    Tpsi_arr = reshape(ein"acmd,bme->abcde"(T_arr, psi_arr), (20, 10))
    @test Tpsi_arr ≈ Tpsi.data

end

# test rrule of act
@timedtestset "test rrule for act ($ix)" for ix in 1:10
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

@timedtestset "test transf_mat constructions ($ix)" for ix in 1:10
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
    
    @test lop_v ≈ lop(v) |> toarray
    @test lopT_v ≈ lop_T(v) |> toarray
end

@timedtestset "dominant vr and vl should be positive definite" for ix in 1:10
    psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
    v = TensorMap(rand, ComplexF64, ℂ^5, ℂ^5)
    lop = transf_mat(psi, psi)
    lop_T = transf_mat_T(psi, psi)

    wr, vr = eigsolve(lop, v, 1)
    wl, vl = eigsolve(lop_T, v, 1)

    @test wr[1] ≈ wl[1]

    vr, vl = vr[1], vl[1]

    # vr, vl is only positive definite up to a phase
    u1, _, v1 = tsvd(vr)
    u2, _, v2 = tsvd(vl)
    @test u1.data ≈ (u1[1,1]/v1[1,1]') * v1.data' 
    @test u2.data ≈ (u2[1,1]/v2[1,1]') * v2.data'

end

@timedtestset "test ovlp and its rrule" for ix in 1:10
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

@timedtestset "test Gamma-Lambda conversion" for ix in 1:10
    psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
    norm_psi, Γ, Λ = lambda_gamma(psi)

    @test ovlp(psi, psi) ≈ norm_psi^2

    @tensor L[-1; -2] := Γ'[-1, 3, 4] * Λ'[3, 2] * Λ[2, 1] * Γ[1, 4, -2]
    @tensor R[-1; -2] := Γ'[3, -2, 4] * Λ'[2, 3] * Λ[1, 2] * Γ[-1, 4, 1]
    Id = id(ℂ^5)
    @test L ≈ Id
    @test R ≈ Id
end

@timedtestset "test mps add" for ix in 1:10
    psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
    phi = TensorMap(rand, ComplexF64, ℂ^6*ℂ^2, ℂ^6)
    
    psi_plus_phi = mps_add(psi, phi)
    @test toarray(psi_plus_phi)[1:5, :, 1:5] ≈ toarray(psi)
    @test toarray(psi_plus_phi)[6:11, :, 6:11] ≈ toarray(phi)
    @test toarray(psi_plus_phi)[1:5, :, 6:11] ≈ zeros(5, 2, 6)
    @test toarray(psi_plus_phi)[6:11, :, 1:5] ≈ zeros(6, 2, 5)
end

@timedtestset "test right_canonical_QR" for ix in 1:10
    chi = 10
    psi = TensorMap(rand, ComplexF64, ℂ^chi*ℂ^2, ℂ^chi)
    Y, psi_R = right_canonical_QR(psi)

    # test the iterated equation
    @tensor rhs[-1, -2; -3] := Y'[-1, 1] * psi_R[1, -2, -3]
    lhs = psi * Y'
    @test lhs ≈ (lhs[1] / rhs[1]) * rhs

    # test the right-canonical relation
    psi_Rtmp = permute(psi_R, (1,), (2, 3))
    @test psi_Rtmp * psi_Rtmp' ≈ id(ℂ^chi)
    # test fidelity
    @test ln_fidelity(psi, psi_R) > -1e-14
end

@timedtestset "test left_canonical_QR" for ix in 1:10
    chi = 10
    psi = TensorMap(rand, ComplexF64, ℂ^chi*ℂ^2, ℂ^chi)
    X, psi_L = left_canonical_QR(psi)
    
    # test the iterated equation
    @tensor lhs[-1, -2; -3] := X[-1, 1] * psi[1, -2, -3]
    rhs = psi_L * X
    @test lhs ≈ (lhs[1] / rhs[1]) * rhs
    
    # test the left canonical relation
    @test psi_L' * psi_L ≈ id(ℂ^chi)
    
    # test the fidelity
    @test ln_fidelity(psi, psi_L) > -1e-14
end
