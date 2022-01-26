@timedtestset "test backwards for energy_lieb_liniger" for ix in 1:10
    ψ_arr = rand(ComplexF64, (3, 3, 3))
    c = 10*rand()
    L = 10*rand()

    function f(arr::Array{ComplexF64, 3})
        ψ = convert_to_cmps(arr)
        return energy_lieb_liniger(ψ, c, L)
    end

    b = FiniteDifferences.grad(central_fdm(5, 1), f, ψ_arr)[1]

    @test f'(ψ_arr) ≈ FiniteDifferences.grad(central_fdm(5, 1), f, ψ_arr)[1]
end