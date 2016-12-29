"""
    `bj(data, nb, nc, nd, nf, nk=1)`

Compute the Box-Jenkins (`nb`,`nc`,`nd`,`nf`,`nk`) model:
    y(t) = z^-`nk`B(z)/F(z)u(t) + C(z)/D(z)e(t)
that minimizes the mean square one-step-ahead prediction error for time-domain IdData `data`.

An initial parameter guess can be provided by adding `x0 = [B; C; D; F]` to the argument list,
where B`, `C`, `F` and `F` are vectors.

To use automatic differentiation add `autodiff=true`.
"""
