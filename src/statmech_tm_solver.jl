module statmech_tm_solver

# Write your package code here.
__precompile__(true)

using LinearAlgebra
using Zygote
using Optim
using Random
using KrylovKit
using TensorKit
using TensorOperations
using JLD2

using TensorKitAD
using ChainRules
using ChainRulesCore
import ChainRulesCore: rrule, frule
import Base: +, -, *, iterate, length, getindex, similar
import TensorKit: leftorth, rightorth

export  act,
        transf_mat,
        transf_mat_T,
        ovlp,
        nonherm_cost_func,
        lambda_gamma,
        left_canonical,
        iTEBD_truncate,
        ln_fidelity,
        free_energy,
        mps_add,
        left_canonical_QR,
        right_canonical_QR,
        entanglement_spectrum,
        entanglement_entropy, 
        expand

export  toarray,
        get_chi,
        get_d,
        arr_to_TensorMap,
        convert_to_tensormap,
        quicksave,
        quickload,
        logsumexp

export  mpo_triangular_AF_ising,
        mpo_triangular_AF_ising_alternative,
        mpo_triangular_AF_ising_adapter,
        mpo_kink_processor,
        mpo_square_ising

export  bimps, 
        A_canonical, 
        B_canonical

export  cmpo,
        get_phy,
        get_vir

export  cmps,
        convert_to_cmps,
        right_canonical,
        K_mat,
        log_ovlp,
        gram_matrix,
        tangent_proj

export  cmpo_ising,
        energy_quantum_ising,
        cmpo_xxz,
        energy_quantum_xxz,
        cmpo_ising_realtime

export  qbimps

export  pseudo_ovlp,
        variational_truncate

export convert_to_array

include("imps.jl")
include("utils.jl")
include("mpo_zoo.jl")
include("bi_direction.jl")
include("cmpo.jl")
include("cmps.jl")
include("cmpo_zoo.jl")
include("bi_direction_q.jl")
include("attempts.jl")
include("TensorKitAD_supp.jl")

end
