@non_differentiable id(V::VectorSpace)
@non_differentiable isomorphism(cod::VectorSpace, dom::VectorSpace)
@non_differentiable domain(W::TensorKit.HomSpace)
@non_differentiable domain(t::AbstractTensorMap)
@non_differentiable domain(t::AbstractTensorMap, i)
@non_differentiable codomain(W::TensorKit.HomSpace)
@non_differentiable codomain(t::AbstractTensorMap)
@non_differentiable codomain(t::AbstractTensorMap, i)

convert_to_array(X::AbstractTensorMap) = convert(Array, X)

function rrule(::typeof(convert_to_array), X::AbstractTensorMap)

    fwd = convert_to_array(X)
    space_X = space(X)

    function convert_to_array_pushback(f̄wd)
        return NoTangent(), TensorMap(f̄wd, space_X)
    end

    return fwd, convert_to_array_pushback
end


"""
    rrule(::typeof(TensorMap), data::A, codom::ProductSpace, dom::ProductSpace) where A <: DenseMatrix

    rrule for TensorMap construction. only did the simplest case.
    FIXME. seems buggy 
"""
function rrule(::typeof(TensorMap), data::A, codom::ProductSpace, dom::ProductSpace) where A <: DenseMatrix
    fwd = TensorMap(data, codom, dom)
    function TensorMap_pushback(f̄wd)
        return NoTangent(), f̄wd.data, NoTangent(), NoTangent()
    end
    return fwd, TensorMap_pushback
end

"""
    rrule(::typeof(tr), A::TensorMap)

    rrule for trace operation.
"""
function rrule(::typeof(tr), A::TensorMap)
    fwd = tr(A)
    function tr_pushback(f̄wd)
        Ā = f̄wd * id(domain(A))
        return NoTangent(), Ā
    end 
    return fwd, tr_pushback
end

"""
    elem_mult(a::AbstractTensorMap,b::AbstractTensorMap)

    Element-wise multiplication between TensorMaps. Copied from TensorKitAD.jl.
"""
function elem_mult(a::AbstractTensorMap,b::AbstractTensorMap)
    dst = similar(a);
    for (k,block) in blocks(dst)
        copyto!(block,blocks(a)[k].*blocks(b)[k]);
    end
    dst
end

"""
    rrule(::typeof(elem_mult), a::AbstractTensorMap, b::AbstractTensorMap)
"""
function rrule(::typeof(elem_mult), a::AbstractTensorMap, b::AbstractTensorMap)
    fwd = similar(a);
    for (k,block) in blocks(fwd)
        copyto!(block,blocks(a)[k].*blocks(b)[k]);
    end
    
    function elem_mult_pushback(f̄wd)
        ā, b̄ = similar(a), similar(b)
        for (k, block) in blocks(f̄wd)
            copyto!(blocks(ā)[k], block .* conj.(blocks(b)[k]))
            copyto!(blocks(b̄)[k], block .* conj.(blocks(a)[k]))
        end

        return NoTangent(), ā, b̄
    end 
    return fwd, elem_mult_pushback
end