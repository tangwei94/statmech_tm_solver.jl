function pseudo_ovlp(psi::TensorMap{ComplexSpace, 2, 1}, phi1::TensorMap{ComplexSpace, 2, 1}, phi2::TensorMap{ComplexSpace, 2, 1}, power_order::Integer)
    ln_ovlp1 = log(ovlp(psi, phi1))
    ln_ovlp2 = log(ovlp(psi, phi2))
    ln_ovlp_tot = ln_ovlp1 + log1p(exp(power_order*(ln_ovlp2-ln_ovlp1))) / power_order
    return ln_ovlp_tot

end

function variational_truncate(phi::TensorMap{ComplexSpace, 2, 1}, chi::Integer)
    _, phi = left_canonical(phi)

    _, psi = iTEBD_truncate(phi, chi)
    psi_arr = toarray(psi)

    function _f(_arr)
        _psi = arr_to_TensorMap(_arr)
        up = norm(ovlp(_psi, phi))^2  
        dn = ovlp(_psi, _psi)
        return -(up/dn) |> real 
    end
    function _g!(_grad, _arr)
        _grad .= gradient(_f, _arr)[1]
    end

    res_f = optimize(_f, _g!, psi_arr, LBFGS(), Optim.Options(show_trace=false, iterations=100))
    psi = Optim.minimizer(res_f) |> arr_to_TensorMap

    return psi
end