@timedtestset "test A_canonical" for ix in 1:10
    psi = bimps(rand, 9, 4)
    T = mpo_triangular_AF_ising_alternative()

    _, psi_a = A_canonical(T, psi)

    # test psi_a.A is right canonical
    A_tmp = permute(psi_a.A, (1, ), (2, 3))
    @test A_tmp * A_tmp' ≈ id(ℂ^9) 
    # fidelity
    @test abs(ln_fidelity(psi_a.A, psi.A)) < 1e-14
    # test the fixed point equation
    @tensor lhs[-1, -2; -3] := psi_a.B[1, 3, 4] * T[-2, 5, 2, 3] * psi_a.A[-1, 2, 1] * psi_a.A'[4, -3, 5]
    @test lhs ≈ (lhs[1] / psi_a.B[1]) * psi_a.B
end


@timedtestset "test B_canonical" for ix in 1:10
    psi = bimps(rand, 9, 4)
    T = mpo_triangular_AF_ising_alternative()

    _, psi_b = B_canonical(T, psi)

    # test psi_b.B is left canonical
    @test psi_b.B' * psi_b.B ≈ id(ℂ^9) 
    # fidelity
    @test abs(ln_fidelity(psi_b.B, psi.B)) < 1e-14
    # test the fixed point equation
    @tensor lhs[-1, -2; -3] := psi_b.A[4, 2, 1] * T[5, -2, 2, 3] * psi_b.B[1, 3, -3] * psi_b.B'[-1, 4, 5]
    @test lhs ≈ (lhs[1] / psi_b.A[1]) * psi_b.A
end