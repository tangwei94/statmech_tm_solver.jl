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

O = mpo_square_ising(0.4)
O = mpo_triangular_AF_ising()
A0 = quickload("ckpt_variational_chi16")
@show nonherm_cost_func(O, A0)
@show free_energy(O, A0)

A = TensorMap(rand, ComplexF64, ℂ^16*ℂ^2, ℂ^16)
_, AL = left_canonical_QR(A)
_, AR = right_canonical_QR(A)

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

W = effective_spect(A*exp(1im*pi/2))
Wmax = findmax(abs.(W))[1] + 0.1

plot(real.(W), imag.(W), seriestype=:scatter, xlims=(-Wmax, Wmax), ylims=(-Wmax, Wmax), size=(500,500))

for ix in 1:20
    fmap_AC, fmap_C = tangent_map(O, AL, AR)

    #shifted_fmap_AC(AC::TensorMap{ComplexSpace, 2, 1}) = fmap_AC(AC) + AC
    #shifted_fmap_C(C::TensorMap{ComplexSpace, 1, 1}) = fmap_C(C) + C

    AC = TensorMap(rand, ComplexF64, ℂ^16*ℂ^2, ℂ^16)
    C = TensorMap(rand, ComplexF64, ℂ^16, ℂ^16)
    _, AC = eigsolve(fmap_AC, AC, 1, :LR)
    _, C = eigsolve(fmap_C, C, 1, :LR)
    AC = AC[1]
    C = C[1]

    #AC = shifted_fmap_AC(AC)
    #C = shifted_fmap_C(C)

    AL, AR = calculate_ALR(AC, C)
    #WAC, WC = effective_spect(AL, AR)
    #Wmax = findmax(abs.(WAC))[1] + 0.1
    #plot(real.(WAC), imag.(WAC), seriestype=:scatter, xlims=(-Wmax, Wmax), ylims=(-Wmax, Wmax), size=(500,500))
    #plot!(real.(WC), imag.(WC), seriestype=:scatter)
    @show nonherm_cost_func(O, AL), nonherm_cost_func(O, AR)
    #@show free_energy(O, AL), free_energy(O, AR)

    lopR = transf_mat(AR, O, AR)
    ER = TensorMap(rand, ComplexF64, ℂ^16*ℂ^2, ℂ^16)
    ws, vs = eigsolve(lopR, ER, 1); @show abs.(ws)

end

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
