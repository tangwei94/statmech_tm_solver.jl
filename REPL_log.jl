# a log of things that I put into the julia REPL
# useful things will be later absorbed into the package 

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

A = TensorMap(rand, ComplexF64, ℂ^4, ℂ^4)
B = TensorMap(rand, ComplexF64, ℂ^4, ℂ^4)
Wvec = rand(4) + im .* rand(4)

L = 1
 
function theta3(a::Number, b::Number, c::Number)
    if a ≈ b ≈ c 
        return 0.5 * exp(L*b) * L^2
    elseif a ≈ b && b != c 
        return -1 * (exp(L*b) - exp(L*c) - L*exp(L*b)*(b-c) ) / (b - c)^2
    elseif b ≈ c && c != a 
        return -1 * (exp(L*c) - exp(L*a) - L*exp(L*c)*(c-a) ) / (c - a)^2
    elseif c ≈ a && a != b 
        return -1 * (exp(L*a) - exp(L*b) - L*exp(L*a)*(a-b) ) / (a - b)^2
    else
        return (a * (exp(L*b) - exp(L*c)) + b * (exp(L*c) - exp(L*a)) + c * (exp(L*a) - exp(L*b))) / ((a-b)*(b-c)*(c-a))
    end
end

for (ix, wx) in enumerate(Wvec)
    for (iy, wy) in enumerate(Wvec)
        for (iz, wz) in enumerate(Wvec)
            θ[ix, iy, iz] = theta3(wx, wy, wz)
            δ[ix, iy, iz] = ComplexF64(ix == iy == iz)
        end
    end
end
θ = Tensor(θ, ℂ^4*ℂ^4*ℂ^4)
δ = TensorMap(δ, ℂ^4, ℂ^4*ℂ^4)

ψ = cmps(rand, 4, 2)
gauge_fixing_proj(ψ, L)

















#########################################################
f1 = p -> (sqrt((p^2 - μ)^2 - 4 * ν^2) - (p^2 - μ)) / 2

Eexact = map(f1, (-100000:100000) .* 2*pi/L ) .* (1 / L) |> sum

@show (E - Eexact) / Eexact
