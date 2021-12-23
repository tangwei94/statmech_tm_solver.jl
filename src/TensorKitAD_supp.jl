@non_differentiable id(V::VectorSpace)
@non_differentiable isomorphism(cod::VectorSpace, dom::VectorSpace)

convert_to_array(X::AbstractTensorMap) = convert(Array, X)

function rrule(::typeof(convert_to_array), X::AbstractTensorMap)

    fwd = convert_to_array(X)
    space_X = space(X)

    function convert_to_array_pushback(f̄wd)
        return NoTangent(), TensorMap(f̄wd, space_X)
    end

    return fwd, convert_to_array_pushback
end

