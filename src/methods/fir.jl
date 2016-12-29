function predict{T1,V1,V2,T2,S}(data::IdDataObject{T1,V1,V2},
  model::PolynomialModel{S,FIR}, Θ::Vector{T2}; ic::Symbol=:zero)

  m = model.orders.nb*ny
  Θic = Θ[1:m]
  a,b,f,c,d = _getpolys(model, Θ[m+1:end])
  return filt(b, a, data.u, reshape(Θic,order(b),ny)) # 10.53 [Ljung1999]
end
