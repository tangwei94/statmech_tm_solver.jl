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