immutable ARX <: OneStepIdMethod
  ic::Symbol
  feedthrough::Bool

  @compat function (::Type{ARX})(ic::Symbol, feedthrough::Bool)
    @assert in(ic, Set([:truncate,:zero]))  string("ic must be either :truncate or :zero")
      new(ic, feedthrough)
  end
end

function ARX(; ic::Symbol=:truncate, feedthrough::Bool=false)
  ARX(ic, feedthrough)
end

function fitmodel{T<:Real}(data::IdDataObject{T}, n::Vector{Int}, method::ARX; kwargs...)
  arx(data, n, method)
end

function arx{T<:Real}(data::IdDataObject{T}, n::Vector{Int}, method::ARX=ARX())
  y,u      = data.y, data.u
  N        = size(y,1)
  na,nb,nk = n
  m        = max(na,nb+nk-1)
  @assert na > -1 string("na must be positive")
  @assert nb > -1 string("nb must be positive")
  @assert nk > -1 string("nk must be positive")
  @assert na+nb > 0 string("na+nb must be greater than zero")

  # estimate model
  x, mse = _arx(data, n, method)

  # Calculate model error
  modelfit = 100*(1 - mse/norm(y[m:N]-mean(y[m:N])))

  a = vcat(ones(T,1),   x[1:na])
  b = vcat(zeros(T,nk), x[na+1:na+nb])
  c = ones(T,1)
  d = ones(T,1)
  f = ones(T,1)
  idinfo = OneStepIdInfo(mse, modelfit, method, n)
  IdDSisoRational(a, b, c, d, f, data.Ts, idinfo)
end

function _arx{T<:Real, V1<:AbstractVector, V2<:AbstractVector}(
    data::IdDataObject{T,V1,V2}, n::Vector{Int},
    method::ARX=method(:truncate, false))
  ic          = method.ic
  feedthrough = method.feedthrough
  y, u     = data.y, data.u
  na,nb,nk = n
  N        = size(y,1)
  m        = max(na, nb)
  md       = max(na, nb+nk-1)
  l        = 2

  theta  = zeros(float(T),nb+na)
  thetam = zeros(float(T),2*m)
  # Compute efficiently for order m
  if ic == :truncate
    col, row, Y = _fill_truncate_ic(data, md, m, n, feedthrough)
  elseif ic == :zero
    col, row, Y = _fill_zero_ic(data, md, m, n, feedthrough)
  end
  thetam[:] = _ls_toeplitz(col, row, Y)
  # remove extra parameters and order correctly
  theta[1:nb]     = thetam[1:l:(nb-1)*l+1]
  theta[nb+1:end] = thetam[l:l:na*l]

  # calculate mse
  y_est = _predict_arx(y, u, na, nb, nk, theta; ic=ic)
  mse   = sumabs2(data.y-y_est)/(N-m+1)
  return theta, mse
end

function predict{T1<:Real, V1<:AbstractVector, V2<:AbstractVector, V3<:AbstractVector}(
    data::IdDataObject{T1,V1,V2}, n::Vector{Int}, x::V3, method::ARX)
  return _predict_arx(data.y, data.u, n..., x; ic=method.ic)
end

function _predict_arx{V1<:AbstractVector, V2<:AbstractVector, V3<:AbstractVector}(
    y::V1, u::V2, na::Int, nb::Int, nk::Int, theta::V3; ic::Symbol=:truncate)
  T = promote_type(eltype(y), eltype(u), eltype(theta))
  m = max(na, nb+nk-1)

  a = append!(zeros(T,1),  theta[1:na])
  b = append!(zeros(T,nk), theta[na+1:na+nb])
  N = length(y)
  y_est = zeros(T,N)
  if ic == :truncate
    # assume y_est = y for t=1:m to find initial states for the filters
    y_est[1:m]   = y[1:m]
    si           = filtic(b,  ones(T,1), filt(b,  ones(T,1), u[m:-1:1]), u[m:-1:1])
    si2          = filtic(-a, ones(T,1), filt(-a, ones(T,1), y[m:-1:1]), y[m:-1:1])
    y_est[m+1:N] = filt(b, ones(T,1), u[m+1:N], si) + filt(-a, ones(T,1), y[m+1:N], si2)
  else  # ic == :zero
    # zero initial conditions
    y_est = filt(b, ones(T,1), u) + filt(-a, ones(T,1), y)
  end
  return y_est
end

function _arx{T<:Real, A1<:AbstractArray, A2<:AbstractArray}(
    data::IdDataObject{T,A1,A2}, na::Matrix{Int}, nb::Matrix{Int}, nk::Matrix{Int};
    method::ARX=ARX())
  @assert size(u,1) == size(y,1) string("Input and output need have the same number of samples")
  y,u = data.y, data.u
  N   = size(y,1)
  ny  = size(y,2)
  nu  = size(u,2)
  @assert size(nb,2) == nu string("nb must have correct size")
  @assert size(nb,1) == ny string("nb must have correct size")
  @assert size(na,2) == ny string("na must have correct size")
  @assert size(na,1) == ny string("na must have correct size")
  @assert size(nk,2) == nu string("nk must have correct size")
  @assert size(nk,1) == ny string("nk must have correct size")

  m = max(maximum(na), maximum(nb))
  md = max(maximum(na), maximum(nb+nk)-1)
  l = nu+1
  theta::Vector{T} = zeros(T,sum(nb)+sum(na))
  thetam::Matrix{T} = zeros(T,l*m,ny)
  # Compute efficiently for order m
  for iy = 1:ny
    m = max(maximum(na[iy,:]), maximum(nb[iy,:]))
    md = max(maximum(na[iy,:]), maximum(nb[iy,:]+nk[iy,:]-ones(similar(nk))), m)
    if method.ic == :truncate
      col, row, Y = _fill_truncate_ic(data, md, m, iy, na, nb, nk)
    elseif method.ic == :zero
      col, row, Y = _fill_zero_ic(data, md, m, iy, na, nb, nk)
    end
    thetam[:,iy] = _ls_toeplitz(col, row, Y)
  end

  # /TODO simulation and prediction for MIMO ARX
  return thetam
end

# Auxilliary methods

function _ls_toeplitz{V<:AbstractVector,M<:AbstractMatrix}(col::M, row::M, Y::V)
  γ = norm(col)
  row /= γ
  col /= γ  # 5γ
  Y   /= γ

  T = full(Toeplitz(col,row))
  Q,R = qr(T)
  return R\(Q.'*Y)
#  return lstoeplitz(col, row, Y)[1]
#  Q,R =  qrtoeplitz(col, row)
#  return R\(Q.'*Y)
end

function _fill_truncate_ic{T<:Real, V1<:AbstractVector, V2<:AbstractVector}(
    data::IdDataObject{T,V1,V2}, md::Int, m::Int, n::Vector{Int},
    feedthrough::Bool=false)
  na,nb,nk = n
  y,u      = data.y, data.u
  ny       = data.ny
  nu       = data.nu
  N        = size(y,1)
  l        = nu+ny

  Y   = feedthrough ? y[md+1:N] - u[md+1:N] :
                      y[md+1:N]
  # trim away delay in u and corresponding elements in y
  ywd = y[md-m+1:N-1]
  uwd = u[md+1-m-(nk-1):N-nk]

  # fill row and col
  col = hcat(-ywd[m:end], uwd[m:end])
  row = zeros(T,1,l*m)
  for i = 1:m
    idx = m+1-i
    row[1,(i-1)*l+1:i*l] = hcat(-ywd[idx], uwd[idx])
  end
  return col, row, Y
end

function _fill_zero_ic{T<:Real, V1<:AbstractVector, V2<:AbstractVector}(
    data::IdDataObject{T,V1,V2}, md::Int, m::Int, n::Vector{Int},
    feedthrough::Bool=false)
  na,nb,nk = n
  y,u      = data.y, data.u
  ny       = data.ny
  nu       = data.nu
  N        = size(y,1)
  l        = 2

  Y = feedthrough ? y - u :
                    y
  # add zero elements in u
  if nk > 0
    ywd = reshape(y[1:end-1],N-1,1)
    uwd = vcat(zeros(T, nk-1,1), reshape(u[1:end-nk],N-1,1))
    Y = Y[2:end]
  else
    ywd = vcat(zeros(T,1,1), reshape(y[1:end-1],N-1,1))
    uwd = reshape(u[1:end-nk],N-1,1)
  end

  # fill col and row
  col = hcat(-ywd, uwd)
  row = hcat(col[1:1,1:l], zeros(T,1,(m-1)*l))
  return col, row, Y
end

function _fill_truncate_ic{M<:AbstractMatrix}(y::M,u::M,md::Int,m::Int,iy::Int,
    na::Matrix{Int},nb::Matrix{Int},nk::Matrix{Int})
  T  = eltype(y)
  N  = size(y,1)
  ny = size(y,2)
  nu = size(u,2)
  l  = nu+ny

  Y   = y[md+1:N,iy][:]
  # trim away delay in u and correspond elements in y
  ywd = y[md-m+1:N-1,:]
  uwd = zeros(T,N-md+m-1,nu)
  for iu = 1:nu
    nkiu = nk[iy,iu]
    println( length(u[md+1-m-(nkiu-1):N-nkiu,iu] ))
    uwd[:,iu] = u[md+1-m-(nkiu-1):N-nkiu,iu]
  end

  # fill row and col
  col = hcat(-ywd[m:end,:], uwd[m:end,:])
  row = zeros(T,1,l*m)
  for i = 1:m
    idx = m+1-i
    row[1,(i-1)*l+1:i*l] = hcat(-ywd[idx,:], uwd[idx,:])
  end
  return col, row, Y
end

function _fill_zero_ic{M<:AbstractMatrix}(y::M,u::M,md::Int,m::Int,iy::Int,
    na::Matrix{Int},nb::Matrix{Int},nk::Matrix{Int})
  T  = eltype(y)
  N  = size(y,1)
  ny = size(y,2)
  nu = size(u,2)
  l  = nu+ny

  Y   = y[1:end,iy][:]
  # add zero elements in y and u
  uwd = zeros(T,N,nu)
  ywd = vcat(zeros(T, ny,1), y[1:end-1,:])
  for iu = 1:nu
    nkiu = nk[iy,iu]
    uwd[:,iu] = vcat(zeros(T, nkiu,1), u[1:end-nkiu,iu])
  end

  # fill col and row
  col = hcat(-ywd[:,:], uwd[:,:])
  row = hcat(col[1,1:l], zeros(T,1,(m-1)*l))
  return col, row, Y
end
