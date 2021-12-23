struct cmpo
    Q::TensorMap{ComplexSpace, 1, 1}
    L::TensorMap{ComplexSpace, 2, 1}
    R::TensorMap{ComplexSpace, 2, 1}
    P::TensorMap{ComplexSpace, 2, 2}
end

@inline get_phy(T::cmpo) = get_chi(T.R)
@inline get_vir(T::cmpo) = get_d(T.R) + 1

@non_differentiable get_phy(T::cmpo)
@non_differentiable get_vir(T::cmpo)