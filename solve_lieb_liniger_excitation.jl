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

using Revise
using statmech_tm_solver

#########################################################

# parameters 
c = 2
μ = 1
L = 50 
d = 1
χ = 20

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

plot(angle.(wvec) .- sign.(angle.(wvec)) .* pi, -real.(wvec), seriestype=:scatter)

sort(real.(wvec))

ixs = -5:5
Ees = []
for ix in ixs
    gauge = :periodic
    p = ix * 2 * pi / L
    χ = get_chi(ψm)
    V0 = TensorMap(rand, ComplexF64, ℂ^(χ*d), ℂ^χ)
    h_lop = lieb_liniger_h_tangent_map(ψm, p, 0, c, L, μ; gauge=gauge)
    n_lop = tangent_map(ψm, L, p; gauge=gauge)

    δ = 1e-8
    Wn, Vn = eigsolve(n_lop, V0, d*χ^2; tol=δ, krylovdim=d*χ^2, ishermitian=true);

    function sqrt_inv_nlop(V0::TensorMap{ComplexSpace, 1, 1})
        tmpf = (Vx, Wx) -> Vx * tr(Vx' * V0) / sqrt(Wx)
        msk = Wn .> δ
        return sum(tmpf.(Vn[msk], Wn[msk])) 
    end

    Ee, Ve = eigsolve(sqrt_inv_nlop ∘ h_lop ∘ sqrt_inv_nlop, V0, 5, :SR; ishermitian=true) 
    Ee = Ee ./ L
    @show ix, Ee
    push!(Ees, Ee)

end

Ees

ks = [0]
Es = [0]
ixs[6]
ΔE = Ees[6][1] - E

for (ix, Ee) in zip(ixs, Ees)
    global ps, Es
    ks = cat(ks, [ix for Ei in Ee]; dims=1)
    Es = cat(Es, (Ee .- E) / ΔE; dims=1)
end
plot(ks, Es, seriestype=:scatter)
xlims!(-5, 5)
ylims!(0, 20)
yticks!(0:20)
