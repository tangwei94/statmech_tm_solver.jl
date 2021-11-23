@timedtestset "test rrule for TensorKit.dot" for ix in 1:10
    psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
    psi_arr = reshape(psi.data, (5, 2, 5))

    tmp1 = gradient(arr->norm(arr)^2, psi_arr)[1]
    tmp2 = gradient(psi->real(dot(psi, psi)), psi)[1]
    tmp2 = reshape(tmp2.data, (5,2,5)) 
    @test tmp1 ≈ conj.(tmp2)
end