# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

using TensorKit
using TensorOperations
using TensorKitAD
using Zygote
using KrylovKit
using LinearAlgebra
using Plots

using Revise
using statmech_tm_solver

AU = TensorMap(rand, ComplexF64, ℂ^12*ℂ^2, ℂ^12)
AD = TensorMap(rand, ComplexF64, ℂ^12*ℂ^2, ℂ^12)
AU_R, AD_R = biorth_right_canonical(AU, AD);
@tensor A_Prod[-1; -2] := AD_R[-1, 2, 1] * AU_R'[1, -2, 2]
AU_L, AD_L = biorth_left_canonical(AU, AD);
AU_L' * AD_L

#############################################

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

A = act(T, AL)
B = act(T_adj, BL)
chi=chi*2
_, AL = left_canonical_QR(A)
_, AR = right_canonical_QR(A)
_, BL = left_canonical_QR(B)
_, BR = right_canonical_QR(B)


"""
function effective_spect(A::TensorMap{ComplexSpace, 2, 1})
    _, AL = left_canonical_QR(A)
    _, AR = right_canonical_QR(A)
    map_AC, map_C = tangent_map_tn(O, AL, AR)
    W, _ = eig(map_C)
    W = diag(W.data)

    return W 
end
function effective_spect(AL::TensorMap{ComplexSpace, 2, 1}, AR::TensorMap{ComplexSpace, 2, 1})
    map_AC, map_C = tangent_map_tn(O, AL, AR)
    WAC, _ = eig(map_AC)
    WAC = diag(WAC.data)

    WC, _ = eig(map_C)
    WC = diag(WC.data)

    return WAC, WC 
end
"""

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

chi=32
AL0 = quickload("AL_chi$(chi)")
AR0 = quickload("AR_chi$(chi)")

lop = transf_mat(AL0, T, AL0)
w, _ = eigsolve(lop, TensorMap(rand, ComplexF64, ℂ^chi*ℂ^2, ℂ^chi), 3)
norm.(w)

mapAC, mapC = tangent_map_tn(T, AL0, AR0);
WC, _ = eig(mapC);
WC = diag(WC.data)
WAC, _ = eig(mapAC);
WAC = diag(WAC.data)

Wmax = findmax(norm.(WAC))[1]
plot(real.(WAC), imag.(WAC), seriestype=:scatter, xlims=(-Wmax, Wmax), ylims=(-Wmax, Wmax), size=(500,500))
plot!(real.(WC), imag.(WC), seriestype=:scatter)

entanglement_spectrum(AL0)
sum(entanglement_spectrum(AR0) .> 1e-5) 
entanglement_entropy(AL0)


A1 = quickload("ckpt_iTEBD_chi32_alternative")
sum(entanglement_spectrum(A1) .> 1e-5)  
entanglement_entropy(A1)


@tensor TA1[-1, -2; -3, -4] := A1'[-3, -1, 1] * A1[-2, 1, -4]

wA1, vRA1 = eig(TA1)
vLA1 = inv(vRA1)
wA1 = diag(wA1.data)
plot(real.(wA1), imag.(wA1), seriestype=:scatter, xlims=(-1, 1), ylims=(-1, 1), size=(500, 500))
thetas = (1:100) * pi / 50
plot!(cos.(thetas), sin.(thetas))
abs_wA1 = abs.(wA1)
sort(abs_wA1)[end-1]

abs_wA1[end-1]

phi = cmps(rand, 6, 4)
psi = cmps(rand, 2, 4)
L = 1.2

function log_fidel(phi::cmps, psi::cmps, L::Real)
    return log_ovlp(phi, psi, L) + log_ovlp(psi, phi, L) - log_ovlp(phi, phi, L) - log_ovlp(psi, psi, L) |> real
end

Q = psi.Q - id(ℂ^get_chi(psi)) * log_ovlp(psi, psi, L) / L / 2
psi = cmps(Q, psi.R)

tangent_proj(phi, psi, L, 1e-3)

tangent_proj(psi, psi, L, 1e-3)

tol = 1e-2
for jx in 1:10
    tol = tol / 10
    do_skip = false
    while ! do_skip
        fidel0 = log_fidel(phi, psi, L)
        Q = psi.Q - id(ℂ^get_chi(psi)) * log_ovlp(psi, psi, L) / L / 2
        psi = cmps(Q, psi.R)
        psi1 = convert_to_cmps(tangent_proj(phi, psi, L, tol))
        step = 0.01
        while log_fidel(phi, step*psi1 + psi, L) < fidel0 && step > 1e-16 
            step /= 2
        end
        do_skip = (step < 1e-16)
        psi = step*psi1 + psi
        @show tol, step, log_fidel(phi, psi, L)
    end
end

cef = diag(vL' * v)

vL1 = vL * Diagonal(1 ./ conj.(cef)) 

adjoint(vL * Diagonal(1 ./ conj.(cef))) - Diagonal(1 ./ cef) * vL'

(vL1' * v) 

@show abs.(vL1' * A_arr * v - Diagonal(w)) .< 1e-14

A_arr - v * Diagonal(w) * vL1'

exp(A_arr) - v * Diagonal(exp.(w)) * vL1'

using BenchmarkTools

@benchmark exp(A_arr)

@benchmark eigen(A_arr)

ψ = cmps(rand, 4, 3)

K = reshape(convert_to_array(K_mat(ψ, ψ)), (16, 16))


A1 = rand(ComplexF64, (4,4))
A2 = rand(ComplexF64, (4,4))
A3 = rand(ComplexF64, (4,4))

@tensor A1[1, 2] * A2[2, 1] * A3[1, 2]
 

psi = cmps(rand, 4, 2)
K = K_mat(psi, psi)

K = permute(K, (2, 1), (4, 3))

D, VR = eigen(K)

VL = inv(VR)'

VL' * K * VR ≈ D
K ≈ VR * D * VL'

D
similar(D)

convert_to_array(psi)

psi = cmps(rand, 2, 3)
chi = get_chi(psi)
Q_arr, R_arr = convert_to_array(psi.Q), convert_to_array(psi.R)
Q_arr = reshape(Q_arr, (chi, 1, chi))
arr = cat(Q_arr, R_arr, dims=2)
