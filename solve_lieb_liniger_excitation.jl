# solve the ground state of Lieb-Liniger model
# also demonstrates the usage of preconditioners in Optim.jl

using TensorKit
using TensorOperations
using TensorKitAD
using KrylovKit
using Zygote
using KrylovKit
using LinearAlgebra
using Plots
using Optim
using ChainRules
using ChainRulesCore
using JLD2
using Printf

using Revise
using statmech_tm_solver

#########################################################

# parameters 
c = 1
μ = 1
L = 50 
d = 1
χ = 16

#########################################################
ψm = quickload("lieb_liniger_c$(c)_mu$(μ)_L$(L)_chi$(χ)") |> convert_to_cmps
ψm = normalize(ψm, L, sym=false)
E = energy_lieb_liniger(ψm, c, L, μ) 

K = K_mat(ψm, ψm)
W, _ = eig(K)
wvec = W.data |> diag

env = finite_env(K_mat(ψm, ψm), L)
env = permute(env, (2, 3), (4, 1))
op_ρ = particle_density(ψm)
op_k = kinetic(ψm)
ρ_value = tr(env * op_ρ) |> real
k_value = tr(env * op_k) |> real
γ = c / ρ_value

#plot(angle.(wvec) .- sign.(angle.(wvec)) .* pi, -real.(wvec), seriestype=:scatter)

#sort(real.(wvec))

Ees = []
kmax = L
for k in -kmax:kmax 
    p = k * 2 * pi / L
    h_lop = lieb_liniger_h_tangent_map(ψm, p, 0, c, L, μ)
    n_lop = tangent_map(ψm, L, p)

    h_lop_mat = zeros(ComplexF64, d*χ^2, d*χ^2)
    n_lop_mat = zeros(ComplexF64, d*χ^2, d*χ^2)
    Vr = TensorMap(zeros, ComplexF64, ℂ^(χ*d), ℂ^χ)
    indices = reshape(1:d*χ^2, (d*χ, χ))
    for ix in 1:d*χ
        for iy in 1:χ
            global Vr, h_lop_mat
            @printf "k, ix, iy = %2d %2d %2d \r" k ix iy
            Vr.data[ix, iy] = 1
            h_lop_mat[:, indices[ix, iy]] = vec(h_lop(Vr).data)
            n_lop_mat[:, indices[ix, iy]] = vec(n_lop(Vr).data)
            Vr.data[ix, iy] = 0
        end
    end
    @assert h_lop_mat ≈ h_lop_mat'
    @assert n_lop_mat ≈ n_lop_mat'

    excitation_data = Dict("hlop" => h_lop_mat, "nlop" => n_lop_mat)
    save("lieb_liniger_excitation_c$(c)_mu$(μ)_L$(L)_chi$(χ)_k$(k).jld2", excitation_data)

    Ee = eigvals(h_lop_mat * inv(Hermitian(n_lop_mat))) ./ L
    push!(Ees, Ee)
end

#ks = [0]
#Es = [0]
#ΔE = Ees[kmax+1][1] - E
#
#for (ix, Ee) in zip(ixs, Ees)
#    global ps, Es
#    ks = cat(ks, [ix for Ei in Ee]; dims=1)
#    Es = cat(Es, (Ee .- E) / ΔE; dims=1)
#end
#plot(ks, real.(Es), seriestype=:scatter)
#xlims!(-5, 5)
#ylims!(0, 20)
#yticks!(0:20)
