function cmpo_ising(Γ::Number)
    Q = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    L = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^1, ℂ^2)
    R = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^1, ℂ^2)
    P = TensorMap(zeros, ComplexF64, ℂ^1*ℂ^2, ℂ^2*ℂ^1)

    Q[1, 2] = Γ
    Q[2, 1] = Γ
    L[1, 1, 1] = 1.0
    L[2, 1, 2] = -1.0
    R[1, 1, 1] = 1.0
    R[2, 1, 2] = -1.0

    return cmpo(Q, L, R, P)
end

function cmpo_xxz(Δ::Number)
    Q = TensorMap(zeros, ComplexF64, ℂ^2, ℂ^2)
    L = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^3, ℂ^2)
    R = TensorMap(zeros, ComplexF64, ℂ^2*ℂ^3, ℂ^2)
    P = TensorMap(zeros, ComplexF64, ℂ^3*ℂ^2, ℂ^2*ℂ^3)

    L[1, 1, 2] = 1.0 
    L[2, 2, 1] = 1.0 
    L[1, 3, 1] = -Δ 
    L[2, 3, 2] = Δ

    R[1, 1, 2] = 1.0
    R[2, 2, 1] = 1.0
    R[1, 3, 1] = Δ
    R[2, 3, 2] = -Δ

    return cmpo(Q, L, R, P)
end