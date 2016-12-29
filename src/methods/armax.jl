
"""
    `armax(data, na, nb, nc, nk=1)`
Compute the ARMAX(`na`,`nb`,`nc`,`nd`) model:
    A(z)y(t) = z^-`nk`B(z)u(t) + C(z)e(t)
that minimizes the mean square one-step-ahead prediction error for time-domain IdData `data`.
An initial parameter guess can be provided by adding `x0 = [A; B; C]` to the argument list, where `A`, `B` and `C` are vectors.
To use automatic differentiation add `autodiff=true`.
"""
