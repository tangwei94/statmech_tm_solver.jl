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

@timedtestset "test act" begin

    for ix in 1:5
        psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
        psi_arr = reshape(psi.data, (5, 2, 5))
        Tpsi = act(T, psi)
        Tpsi_arr = reshape(ein"acmd,bme->abcde"(T_arr, psi_arr), (20, 10))
        @test Tpsi_arr ≈ Tpsi.data
    end

end

# test rrule of act
@timedtestset "test rrule for act" begin
    function f_with_OMEinsum(psi_arr::Array{ComplexF64, 3})
        Tpsi_arr = reshape(ein"acmd,bme->abcde"(T_arr, psi_arr), (10, 2, 10))
        result = ein"abc,abc->"(conj(Tpsi_arr), Tpsi_arr) |> real
        return result[1]
    end
    function f(psi::TensorMap{ComplexSpace, 2, 1})
        Tpsi = act(T, psi)
        return dot(Tpsi, Tpsi) |> real
    end
    
    for ix in 1:10
        psi = TensorMap(rand, ComplexF64, ℂ^5*ℂ^2, ℂ^5)
        psi_arr = reshape(psi.data, (5, 2, 5))
        @test f(psi) ≈ f_with_OMEinsum(psi_arr)
    
        grad_von_OMEinsum = gradient(f_with_OMEinsum, psi_arr)[1]
        grad = reshape(gradient(f, psi)[1].data, (5, 2, 5))
        @test grad_von_OMEinsum ≈ conj.(grad)
    end
end
