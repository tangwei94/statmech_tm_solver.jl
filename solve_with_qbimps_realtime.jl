using Revise
using TensorKit
using statmech_tm_solver
using Printf

Γ = 1.0
T = cmpo_ising_realtime(Γ)

println("========================================================================")

psi = qbimps(rand, 4, 2, 1)
for chi in 8:8:40
    psi = expand(psi, chi, 1e-9)
    println("===========================  chi = $chi   ==============================")
    ix, δ = 0, 42.
    while ix < 250 && δ > 1e-12
        ΛA, psi = A_canonical(T, psi)
        ΛB, psi = B_canonical(T, psi)
        ix += 1
        δ = norm(ΛA - ΛB) / norm(ΛA + ΛB) 
        println("$ix, ΛA: $ΛA, ΛB: $ΛB, δ: $δ")
    end
end
