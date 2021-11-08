function pseudo_ovlp(psi::TensorMap{ComplexSpace, 2, 1}, phi1::TensorMap{ComplexSpace, 2, 1}, phi2::TensorMap{ComplexSpace, 2, 1}, power_order::Integer)
    ln_ovlp1 = log(ovlp(psi, phi1))
    ln_ovlp2 = log(ovlp(psi, phi2))
    ln_ovlp_tot = ln_ovlp1 + log1p(exp(power_order*(ln_ovlp2-ln_ovlp1))) / power_order
    return ln_ovlp_tot
    #ovlp1 = ovlp(psi, phi1)
    #ovlp2 = ovlp(psi, phi2)
    #return (ovlp1^power_order + ovlp2^power_order)^(1.0/power_order) |> real

end

function variational_truncate(phi1::TensorMap{ComplexSpace, 2, 1}, phi2::TensorMap{ComplexSpace, 2, 1}, chi::Integer, power_order::Integer)
    _, phi1 = left_canonical(phi1)
    _, phi2 = left_canonical(phi2)

    #_, psi = iTEBD_truncate(mps_add(phi1, phi2), chi)
    #psi_arr = toarray(psi)
    psi_arr = rand(ComplexF64, chi, 2, chi)

    function _f(_arr)
        _psi = arr_to_TensorMap(_arr)
        up = 2 * pseudo_ovlp(_psi, phi1, phi2, power_order) 
        dn = ovlp(_psi, _psi) |> log
        return (dn - up) |> real 
    end
    function _g!(_grad, _arr)
        _grad .= gradient(_f, _arr)[1]
    end

    res_f = optimize(_f, _g!, psi_arr, LBFGS(), Optim.Options(show_trace=false, iterations=100))
    psi = Optim.minimizer(res_f) |> arr_to_TensorMap

    return psi
end