function pseudo_ovlp(psi::TensorMap{ComplexSpace, 2, 1}, phi1::TensorMap{ComplexSpace, 2, 1}, phi2::TensorMap{ComplexSpace, 2, 1}, power_order::Integer)
    ln_ovlp1 = log(ovlp(psi, phi1))
    ln_ovlp2 = log(ovlp(psi, phi2))
    ln_ovlp_tot = ln_ovlp1 + log1p(exp(power_order*(ln_ovlp2-ln_ovlp1))) / power_order
    return ln_ovlp_tot
end

