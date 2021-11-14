# algorithm 4 in section 5.4 of https://www.annualreviews.org/doi/10.1146/annurev-conmatphys-031016-025507

struct qbimps
    A::TensorMap{ComplexSpace, 2, 1}
    B::cmps
end

function bimps(f, chi::Integer, dp::Integer, dv::Integer)
    A = TensorMap(f, ComplexF64, ℂ^chi*ℂ^dp, ℂ^chi)
    B = cmps(f, chi, dv)
    return qbimps(A, B)
end

function A_canonical(T::TensorMap{ComplexSpace, 2, 2}, psi::qbimps, tol::Float64=1e-15)

end

function B_canonical(T::TensorMap{ComplexSpace, 2, 2}, psi::qbimps, tol::Float64=1e-15)

end

