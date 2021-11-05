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