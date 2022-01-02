# test the "tangent projection" approach in the somewhat trival 3-site uniform mps
# roughly equivalent to the appraoch used in PhysRevB.94.035133 

using TensorKit
using TensorOperations
using TensorKitAD
using Zygote
using KrylovKit
using LinearAlgebra
using Revise
using statmech_tm_solver
using Optim

function fovlp(A::TensorMap{ComplexSpace, 2, 1}, B::TensorMap{ComplexSpace, 2, 1})
    @tensor A[3, 7, 1] * A[1, 8, 2] * A[2, 9, 3] * B'[4, 6, 7] * B'[5, 4, 8] * B'[6, 5, 9]
end

function fidel(A::TensorMap{ComplexSpace, 2, 1}, B::TensorMap{ComplexSpace, 2, 1})
    fovlp(A, B) * fovlp(B, A) / fovlp(A, A) / fovlp(B, B) |> real
end

Afull = 0.1*TensorMap(rand, ComplexF64, ℂ^8*ℂ^2, ℂ^8) 
A = TensorMap(rand, ComplexF64, ℂ^2*ℂ^2, ℂ^2) 
Iph = id(ℂ^2)

A = A / fovlp(A, A)^(1/6)

for ix in 1:75
    @tensor G3[-1, -2, -3; -4, -5, -6] := 
        A[-4, 1, 2] * A'[3, -3, 1] * A[2, 4, -5] * A'[-1, 3, 4] * Iph[-2, -6] +
        A[2, 1, -5] * A'[3, -3, 1] * A'[-1, 3, -6] * A[-4, -2, 2] +
        A'[3, -3, -6] * A[-4, 1, 2] * A'[-1, 3, 1] * A[2, -2, -5] ;

    _, S, _ = tsvd(G3)

    G3_inv = pinv(G3, rtol=1e-12, atol=1e-12);
    G3_inv = permute(G3_inv, (6, 2, 3), (4, 5, 1))

    @tensor A_new[-1, -2; -3] := Afull[6, 7, 4] * Afull[4, 8, 5] * Afull[5, 9, 6] * A'[1, 3, 7] * A'[3, 2, 9] * G3_inv[2, -1, -2, 1, 8, -3]

    fidel0 = fidel(Afull, A)
    coeff = 1.0

    while fidel(Afull, A_new*coeff+A*(1-coeff)) < fidel0
        coeff = 0.5*coeff
    end
    #@show ix, coeff, fidel(Afull, A_new*coeff+A*(1-coeff))

    A = coeff*A_new + A*(1-coeff)
end
@show fidel(Afull, A)
@show A

A2_arr = rand(ComplexF64, (2,2,2))
function f2(X::Array{<:Number, 3})
    X_TensorMap = convert_to_tensormap(X, 2)
    return -fidel(Afull, X_TensorMap)    
end
function g2!(G::Array{<:Number, 3}, X::Array{<:Number, 3})
    G .= gradient(f2, X)[1]
end

res = optimize(f2, g2!, A2_arr,  LBFGS(), Optim.Options(iterations=100) )