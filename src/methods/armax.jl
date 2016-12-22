function predict{T1<:Real,V1,V2,T2<:Real}(
    data::IdDataObject{T1,V1,V2},
    model::PolynomialModel{FullPolynomialOrder,ARMAX},
    Θ::Vector{T2}; ic::Symbol=:zero)
  nb,nf,nk = oeorders(model.orders)
  y,u      = data.y, data.u
  ny,nu    = data.ny, data.nu

  a,b,f,c,d = _getpolys(model, Θ)
  # 10.53 [Ljung1999]
  return filt(b, c, u) + filt(c-a,c,y)
end

function psit{T<:Real,V1,V2}(
    data::IdDataObject{T,V1,V2},
    model::PolynomialModel{FullPolynomialOrder,ARMAX},
    Θ::Vector{T})

  na,nb,nf,nc,nd,nk = orders(model)

  y, u  = data.y, data.u
  m     = max(nf, nb+nk-1)+1
  N     = size(y,1)
  k     = (na+nc)*ny+nb*nu

  y_est     = predict(data, model, Θ)
  a,b,f,c,d = _getpolys(model, Θ)

  ϵ         = y - y_est
  yf        = filt(1, c, y)
  uf        = filt(1, c, u)
  ϵf        = filt(1, c, ϵ)

  Psit      = zeros(T,N,k)

  # a
  row                             = vcat(zeros(T,1,ny), yf[1:end-1,:])
  Psit[1:N,1:na*ny]               = Toeplitz(row, zeros(T,1,ny*na))

  # b
  row                             = vcat(zeros(T,1,nu), uf[1:end-1,:])
  Psit[1:N,na*ny+(1:nb*nu)]       = Toeplitz(row, zeros(T,1,nu*nb))

  # c
  row                             = vcat(zeros(T,1,ny), -ϵf[1:end-1,:])
  Psit[1:N,na*ny+nb*nu+(1:nc*ny)] = Toeplitz(row, zeros(T,1,ny*nc))

  return Psit
end
