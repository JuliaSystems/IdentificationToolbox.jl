
# function predict{T1,V1,V2,T2,S}(data::IdDataObject{T1,V1,V2},
#   model::PolynomialModel{S,OE}, Θ::Vector{T2}; ic::Symbol=:zero)
#   a,b,f,c,d = _getpolys(model, Θ)
#   return filt(b, f, data.u) # 10.53 [Ljung1999]
# end

function psit{T<:Real,V1,V2}(
    data::IdDataObject{T,V1,V2},
    model::PolynomialModel{FullPolynomialOrder{ControlCore.Siso{true}},OE},
    Θ::Vector{T})

  na,nb,nf,nc,nd,nk  = orders(model)

  ny,nu     = data.ny, data.nu
  N         = size(y,1)
  k         = nf*ny+nb*nu

  Θₚ,icbf,icdc,iccda = _split_params(model, Θ, IdOptions())
  a,b,f,c,d          = _getpolys(model, Θₚ)

  w         = filt(b, f, data.u)
  uf        = filt(1, f, data.u)
  wf        = filt(1, f, w)

  Psit      = zeros(T,N,k)

  # b
  col                       = vcat(zeros(T,1,nu), uf[1:end-1,:])
  Psit[1:N,1:nu*nb]         = Toeplitz(col, zeros(T,1,nu*nb))

  # f

  col                       = vcat(zeros(T,1,ny), -wf[1:end-1,:])
  Psit[1:N,nb*nu+(1:ny*nf)] = Toeplitz(col, zeros(T,1,ny*nf))

  return Psit
end

"""
    `oe(data, nb, nf, nk=1)`

Compute the OE(nb`,`nf`,`nd`) model:
    F(z)y(t) = z^-`nk`B(z)u(t) + F(z)e(t)
that minimizes the mean square one-step-ahead prediction error for time-domain IdData `data`.

An initial parameter guess can be provided by adding `x0 = [B, F]` to the argument list,
where `B` and `F` are vectors.

To use automatic differentiation add `autodiff=true`.
"""
