function cmpo_ising(Γ::Float64)
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
