using TensorKit
using statmech_tm_solver

chi = 64
psi = qbimps(rand, chi, 2, 3)
Δ = 1.0 

T = cmpo_xxz(Δ)
println("========================================================================")

ix, δ = 0, 42.
while ix < 50 && δ > 1e-12
    ΛA, psi = A_canonical(T, psi)
    ΛB, psi = B_canonical(T, psi)
    ix += 1
    δ = norm(ΛA - ΛB) / norm(ΛA + ΛB) 
    println("$ix, ΛA: $ΛA, ΛB: $ΛB, δ: $δ")
end

sA = entanglement_spectrum(psi.A)
sB = entanglement_spectrum(psi.B)

io = open("result_qbimps_chi$(chi)_xxz_Delta$(Δ).txt", "w")
for s in sA
    print(io, "$s ")
end
println(io, " ")
for s in sB
    print(io, "$s ")
end
println(io, " ")
close(io)