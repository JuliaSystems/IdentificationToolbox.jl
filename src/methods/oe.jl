
function _getoepolys{T<:Real,S,V}(
  model::PolynomialModel{FullPolynomialOrder{S},V}, Θ::Vector{T})
  nb,nf,nk = oeorders(model.orders)
  ny,nu    = model.ny, model.nu

  m  = nb*nu+nf*ny
  x  = reshape(Θ, m, ny)
  xb = _blocktranspose(view(x, 1:nb*nu, 1:ny), ny, nu, nb)
  xf = _blocktranspose(view(x, nb*nu+(1:nf*ny), 1:ny), ny, ny, nf)

  # zero pad vectors
  b = PolyMatrix(vcat(zeros(T,ny,nu), xb), (ny,nu))
  f = PolyMatrix(vcat(eye(T,ny),      xf), (ny,ny))

  return b, f
end

oeorders(n::FullPolynomialOrder) = (n.nb, n.nf, n.nk)

function predict{T1<:Real,V1,V2,T2<:Real,S}(
    data::IdDataObject{T1,V1,V2},
    model::PolynomialModel{FullPolynomialOrder{S},OE},
    Θ::Vector{T2}; ic::Symbol=:zero)
  nb,nf,nk = oeorders(model.orders)
  y,u      = data.y, data.u
  ny,nu    = data.ny, data.nu

  a,b,f,c,d = _getpolys(model, Θ)
  # 10.53 [Ljung1999]
  return filt(b, f, u)
  #est(y,u,b,f,nk,ic)
end

function psit{T<:Real,V1,V2,S}(
    data::IdDataObject{T,V1,V2},
    model::PolynomialModel{FullPolynomialOrder{S},OE},
    Θ::Vector{T}, l::Int)

  y, u      = data.y, data.u
  nb,nf,nk  = oeorders(model.orders)
#  m         = max(nf, nb+nk-1)+1
  N         = size(y,1)
  k         = nf*ny+nb*nu

  a,b,f,c,d = _getpolys(model, Θ)

  w         = filt(b, f, u)
  uf        = filt(1, f, u)
  wf        = filt(1, f, w)

  Psit      = zeros(T,N,k)

  # b
  row                       = vcat(zeros(T,1,nu), uf[1:end-1,:])
  Psit[1:N,1:nu*nb]         = Toeplitz(row, zeros(T,1,nu*nb))

  # f
  row                       = vcat(zeros(T,1,ny), -wf[1:end-1,:])
  Psit[1:N,nb*nu+(1:ny*nf)] = Toeplitz(row, zeros(T,1,ny*nb))

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
function oe{T<:Real}(data::IdDataObject{T}, nb::Int, nf::Int, nk::Int=1,
      x0::AbstractArray = vcat(init_cond(data.y, data.u, na, nb, nc)...); kwargs...)
  N = size(data.y, 1)
  m = max(nf, nb+nk-1)+1
  n = [nb,nf,nk]
  k = nf + nb

  # detect input errors
  any(n .< 0)     && error("nb, nf, nk must be nonnegative integers")
  m>N             && error("Not enough datapoints to fit OE($nb,$nf,$nk) model")
  length(x0) != k && error("Used initial guess of length $(length(x0)) for OE model with $k parameters")

  pem(data, n, x0, OE(kwargs...))
end
