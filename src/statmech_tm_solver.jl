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
        ln_free_energy,
        mps_add

export  toarray,
        get_chi,
        arr_to_TensorMap

export  pseudo_ovlp,
        variational_truncate

include("imps.jl")
include("utils.jl")
include("new_method.jl")

end
