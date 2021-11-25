function rrule(::typeof(dot), t1::TensorMap{ComplexSpace}, t2::AbstractTensorMap{ComplexSpace})
    fwd = dot(t1, t2)

    function dot_pushback(f̄wd)
        @tensor t̄1[-1, -2; -3] := conj(t2[-1, -2, -3])*conj(f̄wd)
        @tensor t̄2[-1, -2; -3] := conj(t1[-1, -2, -3])*f̄wd
        return NoTangent(), t̄1, t̄2
    end
    return fwd, dot_pushback
end

function toarray(t::TensorMap{ComplexSpace})
    @inline tuplejoin(x, y) = (x..., y...)
    t_dims = tuplejoin(dims(codomain(t)), dims(domain(t)))
    return reshape(t.data, t_dims)
end

@inline get_chi(psi::TensorMap{ComplexSpace, 2, 1}) = dim(domain(psi))
@inline get_d(psi::TensorMap{ComplexSpace, 2, 1}) = dim(codomain(psi)) ÷ dim(domain(psi))

function arr_to_TensorMap(arr::Array{ComplexF64, 3})
    chi, d, _ = size(arr)
    return TensorMap(arr, ℂ^chi*ℂ^d, ℂ^chi)
end
function rrule(::typeof(arr_to_TensorMap), arr::Array{ComplexF64, 3})
    chi, d, _ = size(arr)
    fwd = TensorMap(arr, ℂ^chi*ℂ^d, ℂ^chi)
    function arr_to_TensorMap_pushback(f̄wd)
        return NoTangent(), conj.(reshape(f̄wd.data, (chi, d, chi)))
    end
    return fwd, arr_to_TensorMap_pushback
end

"""
    quicksave(name::String, psi::TensorMap{ComplexSpace, 2, 1})

    Save the mps local tensor `psi` to a `.jld2` file.
"""
function quicksave(name::String, psi::TensorMap{ComplexSpace, 2, 1})
    # todo: get convert(Dict, t::AbstractTensorMap) in the manual to work with JLD2?
    chi, d = get_chi(psi), get_d(psi)
    psi_dict = Dict("chi" => chi,
                    "d" => d,
                    "data" => psi.data
    )

    save("$(name).jld2", psi_dict)
end

"""
    quickload(name::String)

    load the mps local tensor `psi` from a `.jld2` file.
"""
function quickload(name::String)
    psi_dict = load("$(name).jld2")
    chi, d = psi_dict["chi"], psi_dict["d"]
    return TensorMap(psi_dict["data"], ℂ^chi*ℂ^d, ℂ^chi)
end