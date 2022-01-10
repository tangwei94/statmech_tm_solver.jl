using TensorKit
using TensorOperations
using TensorKitAD
using Zygote
using KrylovKit
using LinearAlgebra
using Plots

using Revise
using statmech_tm_solver

T = mpo_triangular_AF_ising()
T_adj = reshape(T.data, (2, 2, 2, 2))
T_adj = permutedims(conj.(T_adj), (1, 3, 2, 4))
T_adj = TensorMap(T_adj, ℂ^2*ℂ^2, ℂ^2*ℂ^2)

chi = 4
A = TensorMap(rand, ComplexF64, ℂ^chi*ℂ^2, ℂ^chi)
A = act(T, A); chi=chi*2
_, AL = left_canonical_QR(A)
_, AR = right_canonical_QR(A)
BL, BR = AL, AR

for ix in 1:3
    A = act(T, AL)
    B = act(T_adj, BL)
    chi=chi*2
    _, AL = left_canonical_QR(A)
    _, AR = right_canonical_QR(A)
    _, BL = left_canonical_QR(B)
    _, BR = right_canonical_QR(B)

    flag = 1

    for ix in 1:20
        fmap_AC, fmap_C = tangent_map(T, AL, AR)

        AC = TensorMap(rand, ComplexF64, ℂ^chi*ℂ^2, ℂ^chi)
        C = TensorMap(rand, ComplexF64, ℂ^chi, ℂ^chi)
        _, AC = eigsolve(fmap_AC, AC, 1, :LR) 
        AC = AC[1]
        _, C = eigsolve(fmap_C, C, 1, :LR) 
        C = C[1]

        AL1, AR1, ϵL, ϵR = calculate_ALR(AC, C)

        #WAC, WC = effective_spect(AL, AR)
        #Wmax = findmax(abs.(WAC))[1] + 0.1
        #plot(real.(WAC), imag.(WAC), seriestype=:scatter, xlims=(-Wmax, Wmax), ylims=(-Wmax, Wmax), size=(500,500))
        #plot!(real.(WC), imag.(WC), seriestype=:scatter)

        flag1 = nonherm_cost_func(T, AL1)
        if flag1 > flag
            flag = 1
            break
        else 
            flag = flag1
            AL, AR = AL1, AR1
        end

        @show nonherm_cost_func(T, AL), nonherm_cost_func(T, AR)
        @show ϵL, ϵR
        @show free_energy(T, AL), free_energy(T, AR)
    end

    for ix in 1:20
        fmap_BC, fmap_C = tangent_map(T_adj, BL, BR)

        BC = TensorMap(rand, ComplexF64, ℂ^chi*ℂ^2, ℂ^chi)
        C = TensorMap(rand, ComplexF64, ℂ^chi, ℂ^chi)
        λBC, BC = eigsolve(fmap_BC, BC, 1, :LR) 
        BC = BC[1]
        λC, C = eigsolve(fmap_C, C, 1, :LR) 
        C = C[1]

        BL1, BR1, ϵL, ϵR = calculate_ALR(BC, C)

        #WAC, WC = effective_spect(AL, AR)
        #Wmax = findmax(abs.(WAC))[1] + 0.1
        #plot(real.(WAC), imag.(WAC), seriestype=:scatter, xlims=(-Wmax, Wmax), ylims=(-Wmax, Wmax), size=(500,500))
        #plot!(real.(WC), imag.(WC), seriestype=:scatter)

        flag1 = nonherm_cost_func(T_adj, BL1)
        if flag1 > flag
            flag = 1
            break
        else 
            flag = flag1
            BL, BR = BL1, BR1
        end

        @show nonherm_cost_func(T_adj, BL), nonherm_cost_func(T_adj, BR)
        @show free_energy(T_adj, BL), free_energy(T_adj, BR)
    end

    @show ovlp(BL, act(T, AL)) / ovlp(BL, AL) |> log

    quicksave("AL_chi$(chi)", AL)
    quicksave("AR_chi$(chi)", AR)
    quicksave("BL_chi$(chi)", BL)
    quicksave("BR_chi$(chi)", BR)
end