using TensorKit
using TensorOperations
using TensorKitAD
using Zygote
using KrylovKit
using LinearAlgebra
using Plots
using Optim

using Revise
using statmech_tm_solver

T = cmpo_ising(1.0)
ψ = cmps(T.Q, T.R)

beta = 128
Fexact = -1.2732475342538678 
conv_meas = 1e-4
fidel=-1

log_ovlp(ψ, ψ, beta)

for chi in [2; 4; 8; 12; 16; 20; 24; 28; 32]
    global conv_meas, fidel, conv_meas
    @show chi
    for ix in 1:100
        Tψ = act(T, ψ)
        status, fidel, ψ1 = compress(Tψ, chi, beta; tol=conv_meas)
        conv_meas1 = convergence_measure(T, ψ1, beta)
        @show fidel, conv_meas
        if status
            ψ = ψ1
            conv_meas = conv_meas1 
            f = -(log_ovlp(ψ, act(T, ψ), beta) - log_ovlp(ψ, ψ, beta)) / beta |> real
            @show f, (f - Fexact) / Fexact, conv_meas
        else
            break
        end
    end
    f = -(log_ovlp(ψ, act(T, ψ), beta) - log_ovlp(ψ, ψ, beta)) / beta |> real
    @show chi, get_chi(ψ), f, (f - Fexact) / Fexact
    @show conv_meas

    @show "self-iterating..."
    for ix in 1:50
        print("$(ix) \r")
        Tψ = act(T, ψ)
        status, fidel, ψ1 = compress(Tψ, chi, beta; tol=conv_meas, init=ψ)
        conv_meas1 = convergence_measure(T, ψ1, beta)
        if status && fidel > -conv_meas1 * 0.99
            ψ = ψ1
            conv_meas = conv_meas1 
            f = -(log_ovlp(ψ, act(T, ψ), beta) - log_ovlp(ψ, ψ, beta)) / beta |> real
            @show f, (f - Fexact) / Fexact, fidel, conv_meas
        else
            break
        end
        f = -(log_ovlp(ψ, act(T, ψ), beta) - log_ovlp(ψ, ψ, beta)) / beta |> real
    end

end