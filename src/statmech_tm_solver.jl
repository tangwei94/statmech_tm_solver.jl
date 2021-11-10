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

using ChainRules
using ChainRulesCore
import ChainRulesCore: rrule, frule

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
        right_canonical_QR

export  toarray,
        get_chi,
        arr_to_TensorMap

export  mpo_triangular_AF_ising,
        mpo_triangular_AF_ising_alternative,
        mpo_square_ising

export  pseudo_ovlp,
        variational_truncate

include("imps.jl")
include("utils.jl")
include("mpo_zoo.jl")
include("new_method.jl")

end
