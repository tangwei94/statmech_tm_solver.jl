
function act(T::TensorMap{ComplexSpace, 2, 2}, psi::TensorMap{ComplexSpace, 2, 1})
    chi = dim(psi.dom)
    fuse_tensor = isomorphism(ℂ^(chi*2), ℂ^2*ℂ^chi)

    #Tpsi = @ncon((T, psi, fuse_tensor, fuse_tensor'), ([3, -2, 2, 5], [1, 2, 4], [-1, 3, 1], [5, 4, -3]))
    #Tpsi = permute(Tpsi, (1, 2), (3,))
    @tensor Tpsi[-1, -2; -3] := T[3, -2, 2, 5] * psi[1, 2, 4] * fuse_tensor[-1, 3, 1] * fuse_tensor'[5, 4, -3]
    return Tpsi
end

function rrule(::typeof(act), T::TensorMap{ComplexSpace, 2, 2}, psi::TensorMap{ComplexSpace, 2, 1})
    chi = dim(psi.dom)
    fuse_tensor = isomorphism(ℂ^(chi*2), ℂ^2*ℂ^chi)

    @tensor Tpsi[-1, -2; -3] := T[3, -2, 2, 5] * psi[1, 2, 4] * fuse_tensor[-1, 3, 1] * fuse_tensor'[5, 4, -3]

    fwd = Tpsi
    function act_pushback(f̄wd)
        #psi_pushback = @ncon((f̄wd, fuse_tensor, fuse_tensor', T), 
        #    ([1,4,2], [1, 3, -1], [5, -3, 2], [3, 4, -2, 5])
        #)
        #psi_pushback = permute(psi_pushback, (1, 2), (3,))
        @tensor psi_pushback[-1, -2; -3] := f̄wd[1,4,2] * fuse_tensor[1,3,-1] * fuse_tensor'[5,-3,2] * T[3,4,-2,5]
        return NoTangent(), NoTangent(), psi_pushback
    end
    return fwd, act_pushback
end

function transf_mat(psi::TensorMap{ComplexSpace, 2, 1}, phi::TensorMap{ComplexSpace, 2, 1})
    function lop(v::TensorMap{ComplexSpace, 1, 1})
        @tensor result[-2; -1] := psi'[1, -1, 3] * phi[-2, 3, 2] * v[2, 1]
        return result
    end
    return lop
end

function transf_mat_T(psi::TensorMap{ComplexSpace, 2, 1}, phi::TensorMap{ComplexSpace, 2, 1})
    function lop_T(v::TensorMap{ComplexSpace, 1, 1})
        @tensor result[-1; -2] := psi'[-1, 1, 3] * phi[2, 3, -2] * v[1, 2]
        return result
    end
    return lop_T
end