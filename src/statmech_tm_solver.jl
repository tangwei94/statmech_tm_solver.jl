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
using FiniteDifferences

using ChainRules
using ChainRulesCore
import ChainRulesCore: rrule, frule

export  act,
        transf_mat,
        transf_mat_T

export  toarray

include("imps.jl")
include("utils.jl")

end
