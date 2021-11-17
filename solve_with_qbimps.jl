using Revise
using TensorKit
using statmech_tm_solver
using Printf

Δ = -0.5 
T = cmpo_xxz(Δ)
println("========================================================================")

psi = qbimps(rand, 2, 2, 3)
for chi in [2; 4; 8; 16; 32; 64; 128]
    psi = expand(psi, chi)
    println("===========================  chi = $chi   ==============================")
    ix, δ = 0, 42.
    while ix < 100 && δ > 1e-12
        ΛA, psi = A_canonical(T, psi)
        ΛB, psi = B_canonical(T, psi)
        ix += 1
        δ = norm(ΛA - ΛB) / norm(ΛA + ΛB) 
        println("$ix, ΛA: $ΛA, ΛB: $ΛB, δ: $δ")
        println("energy density: ", energy_quantum_xxz(psi.A, Δ))
    end

    sA = entanglement_spectrum(psi.A)
    sB = entanglement_spectrum(psi.B)

    io = open(@sprintf("result_qbimps_chi%d_xxz_Delta%.2f.txt", chi, Δ), "w")
    for s in sA
        print(io, "$s ")
    end
    println(io, " ")
    for s in sB
        print(io, "$s ")
    end
    println(io, " ")
    close(io)
end
