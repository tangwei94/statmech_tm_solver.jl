function rrule(::typeof(dot), t1::AbstractTensorMap{ComplexSpace}, t2::AbstractTensorMap{ComplexSpace})
    fwd = dot(t1, t2)

    function dot_pushback(f̄wd)
        t̄1 = f̄wd' * t2
        t̄2 = f̄wd * t1 
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

@non_differentiable get_chi(psi::TensorMap{ComplexSpace, 2, 1})
@non_differentiable get_d(psi::TensorMap{ComplexSpace, 2, 1})

function arr_to_TensorMap(arr::Array{ComplexF64, 3})
    chi, d, _ = size(arr)
    return TensorMap(arr, ℂ^chi*ℂ^d, ℂ^chi)
end
function rrule(::typeof(arr_to_TensorMap), arr::Array{ComplexF64, 3})
    chi, d, _ = size(arr)
    fwd = TensorMap(arr, ℂ^chi*ℂ^d, ℂ^chi)
    function arr_to_TensorMap_pushback(f̄wd)
        return NoTangent(), reshape(f̄wd.data, (chi, d, chi))
    end
    return fwd, arr_to_TensorMap_pushback
end

*(x::ComplexSpace) = x

"""
    convert_to_tensormap(arr::Array{<:Complex}, d_dom::Integer) -> TensorMap

    Given the dimension of the domain, convert an array to the tensormap.
    FIXME: fix vector cases. 
"""
function convert_to_tensormap(arr::Array{<:Complex}, d_dom::Integer)
    size_arr = size(arr)
    d_arr = length(size_arr)
    dom = ifelse(d_dom > 0, prod(map(x->ℂ^x, size_arr[1:d_dom])), ProductSpace{ComplexSpace, 0})
    codom = ifelse(d_arr > d_dom, prod(map(x->ℂ^x, size_arr[d_dom+1:end])), ProductSpace{ComplexSpace, 0})
    return TensorMap(arr, dom, codom)
end

"""
    rrule(typeof(convert_to_tensormap), arr::Array{<:Complex}, d_dom::Integer)

    The rrule for function `convert_to_tensormap`.
"""
function rrule(::typeof(convert_to_tensormap), arr::Array{<:Complex}, d_dom::Integer)
    fwd = convert_to_tensormap(arr, d_dom)

    function convert_to_tensormap_pushback(f̄wd)
        return NoTangent(), convert(Array, f̄wd), NoTangent()
    end
    return fwd, convert_to_tensormap_pushback
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

"""
    logsumexp(w)

Calculation of logsumexp.
"""
function logsumexp(w::Array{<:Number, 1})
    u = maximum(norm.(w))
    t = 0.0
    for i = 1:length(w)
        t += exp(w[i]-u)
    end
    u + log(t)
end