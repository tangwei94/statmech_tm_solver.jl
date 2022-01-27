@timedtestset "test rrule for TensorKit.dot" for ix in 1:10
    psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
    psi_arr = reshape(psi.data, (5, 2, 5))

    tmp1 = gradient(arr->norm(arr)^2, psi_arr)[1]
    tmp2 = gradient(psi->real(dot(psi, psi)), psi)[1]
    tmp2 = reshape(tmp2.data, (5,2,5)) 
    @test tmp1 ≈ tmp2
end

@timedtestset "test element-wise multiplication and its rrule" for ix in 1:10
    psi = TensorMap(rand, ComplexF64, ℂ^3*ℂ^2, ℂ^3)
    phi = TensorMap(rand, ComplexF64, ℂ^3*ℂ^2, ℂ^3)
    psi_arr = convert_to_array(psi)
    phi_arr = convert_to_array(phi)

    @test elem_mult(psi, phi) ≈ convert_to_tensormap(psi_arr .* phi_arr, 2)

    norm(elem_mult(psi, phi))

    function f(arr::Array{ComplexF64, 3})
        a = convert_to_tensormap(arr, 2)
        return norm(elem_mult(a, phi))
    end

    f(psi_arr)
    @test f'(psi_arr) ≈ FiniteDifferences.grad(central_fdm(5, 1), f, psi_arr)[1]

end