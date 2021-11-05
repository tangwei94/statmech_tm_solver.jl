using Revise
using statmech_tm_solver
using Test
using TestExtras

using TensorKit
using OMEinsum
using FiniteDifferences
using Zygote


@testset "imps.jl" begin
    include("test_imps.jl")
end

@testset "utils.jl" begin
    include("test_utils.jl")
end
