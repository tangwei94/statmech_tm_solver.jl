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
using Tullio
using LoopVectorization

using TensorKitAD
using ChainRules
using ChainRulesCore
import ChainRulesCore: rrule, frule
import Base: +, -, *, iterate, length, getindex, similar
import TensorKit: leftorth, rightorth
import LinearAlgebra: normalize, dot, ldiv!

# imps.jl
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
        expand,
        tangent_map_tn, 
        tangent_map, 
        calculate_ALR,
        biorth_right_canonical, 
        biorth_left_canonical

# utils.jl
export  toarray,
        get_chi,
        get_d,
        arr_to_TensorMap,
        convert_to_tensormap,
        quicksave,
        quickload,
        logsumexp

# mpo_zoo.jl
export  mpo_triangular_AF_ising,
        mpo_triangular_AF_ising_alternative,
        mpo_triangular_AF_ising_adapter,
        mpo_kink_processor,
        mpo_square_ising

# bi_direction.jl
export  bimps, 
        A_canonical, 
        B_canonical

# cmpo.jl
export  cmpo,
        get_phy,
        get_vir

# cmps.jl
export  cmps,
        get_matrices,
        convert_to_cmps,
        right_canonical,
        K_mat,
        log_ovlp,
        convergence_measure,
        finite_env,
        kinetic,
        particle_density,
        pairing,
        point_interaction,
        optimize_conv_meas,
        compress,
        truncation_check,
        gauge_fixing_proj,
        tangent_map,
        preconditioner,
        precond_grad, 
        lieb_liniger_h_tangent_map

# cmpo_zoo.jl
export  cmpo_ising,
        cmpo_xy,
        cmpo_xz,
        energy_quantum_ising,
        cmpo_xxz,
        energy_quantum_xxz,
        cmpo_ising_realtime,
        energy_lieb_liniger,
        energy_pairing_boson

# bi_direction_q.jl
export  qbimps

# TensorKitAD_supp.jl
export  convert_to_array,
        elem_mult

include("imps.jl")
include("utils.jl")
include("mpo_zoo.jl")
include("bi_direction.jl")
include("cmpo.jl")
include("cmps.jl")
include("cmpo_zoo.jl")
include("bi_direction_q.jl")
include("TensorKitAD_supp.jl")

end
