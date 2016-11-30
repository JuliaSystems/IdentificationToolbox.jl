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

arx{T<:Real}(data::IdDataObject{T}, n::Vector{Int}, method::ARX=ARX()) =
  arx(data, n[1], n[2], n[3], method)

function arx{T<:Real}(data::IdDataObject{T}, na::Int, nb::Int,
    nk::Vector{Int}, method::ARX=ARX())
  y,u = data.y, data.u
  ny  = data.ny
  nu  = data.nu
  N   = size(y,1)
  m   = _arxordercheck(na,nb,nk)

  # estimate model
  x, mse = _arx(data, na, nb, nk; method=method)

  # Calculate model error 100*(1 - mse[i]/cov(y[m:N,i]))
  modelfit = [100*(1 - mse[i]/cov(y[m:N,i])) for i in 1:ny]

  idinfo = OneStepIdInfo(mse, modelfit, method, vcat([na,nb],nk))

  A = view(x, 1:(na+1)*ny, :)
  B = view(x, (na+1)*ny+1:size(x,1), :)
  C = speye(T,ny)
  D = speye(T,ny)
  F = speye(T,ny)
  IdMFD(A, B, C, D, F, data.Ts, idinfo, ny)
end

function _arxordercheck(na,nb,nk)
  @assert na > -1       string("na must be positive")
  @assert nb > -1       string("nb must be positive")
  @assert all(nk .> -1) string("nk must be positive")
  @assert na+nb > 0     string("na+nb must be greater than zero")
  max(na,nb+maximum(nk)-1)
end

function arx{T<:Real, V1<:AbstractVector, V2<:AbstractVector}(
  data::IdDataObject{T,V1,V2}, na::Int, nb::Int,
    nk::Int, method::ARX=ARX())
  y,u = data.y, data.u
  N   = size(y,1)
  m   = _arxordercheck(na,nb,nk)

  # estimate model
  x, mse = _arx(data, na, nb, nk; method=method)

  # Calculate model error 100*(1 - mse[i]/cov(y[m:N,i]))
  modelfit = 100*(1 - mse/cov(y[m:N]))

  idinfo = OneStepIdInfo(mse, modelfit, method, [na,nb,nk])

  a = vcat(ones(T,1),   x[1:na])
  b = vcat(zeros(T,nk), x[na+1:na+nb])
  c = ones(T,1)
  d = ones(T,1)
  f = ones(T,1)
  IdMFD(a, b, c, d, f, data.Ts, idinfo)
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
    data::IdDataObject{T,A1,A2}, na::Int, nb::Int, nk::Int;
    method::ARX=ARX())
    return _arx(data, na, nb, [nk]; method=method)
end

function _arx{T<:Real, A1<:AbstractArray, A2<:AbstractArray}(
    data::IdDataObject{T,A1,A2}, na::Int, nb::Vector{Int}, nk::Vector{Int};
    method::ARX=ARX())
    return _arx(data, [na], nb, nk; method=method)
end

function _arx{T<:Real, A1<:AbstractArray, A2<:AbstractArray}(
    data::IdDataObject{T,A1,A2}, na::Vector{Int}, nb::Int, nk::Int;
    method::ARX=ARX())
    return _arx(data, na, [nb], [nk]; method=method)
end

function _arx{T<:Real, A1<:AbstractArray, A2<:AbstractArray}(
    data::IdDataObject{T,A1,A2}, na::Int, nb::Int, nk::Vector{Int};
    method::ARX=ARX())
  y,u = data.y, data.u
  N   = size(y,1)
  ny  = data.ny
  nu  = data.nu
  feedthrough = method.feedthrough
  @assert size(u,1)  == N "Input and output need have the same number of samples"
#  @assert length(nb) == nu "nb must have correct size"
#  @assert length(na) == ny "na must have correct size"
  @assert length(nk) == nu "nk must have correct size"
  @assert !feedthrough || ny == nu "Only square B matrices with feedthrough supported"
  @assert !any(nk .< 0) "Negative delays present"

  # handle variable nk
  if any(nk .== 0)
    ye = vcat(y[:,:], zeros(T,1,ny))
    ue = vcat(zeros(T,1,nu), u)
    nk += ones(nu)
  else
    ye = y[:,:]
    ue = u[:,:]
  end
  Nk = maximum(nk)
  ut = zeros(T,size(ue)...)
  for i = 1:nu
    if nk[i] > 1
      ut[nk[i]:end,i] = ue[1:end-nk[i]+1,i]
    else
      ut[:,i] = ue[:,i]
    end
  end
  datat = iddata(ye,ut)

  # assume nk = 1 for all input output pairs

  m = max(maximum(na), maximum(nb))
  md = max(maximum(na), maximum(nb)-1)
  l = nu+ny
  Θ::Matrix{T} = zeros(T,sum(nb)+sum(na),ny)
#  thetam::Matrix{T} = zeros(T,l*m,ny)
  # Compute efficiently for order m

  if method.ic == :truncate
    Φ, Y = _fill_truncate_ic(datat, md, m, feedthrough=feedthrough)
  elseif method.ic == :zero
    Φ, Y = _fill_zero_ic(datat, md, m, feedthrough=feedthrough)
  end
  Θ = _ls_toeplitz(Φ, Y)
  mse = _mse(Θ, Φ, Y)

  # get Θ in the form Θ = [I A₁ᵀ A₂ᵀ… B₁ᵀ B₂ᵀ …]ᵀ
  Θᵣ = zeros(T, (nb+na+1)*ny,ny)
  Θᵣ[1:ny,1:ny] = eye(T,ny)
  for k = 1:na
    idx = (k-1)*2ny+(1:ny)
    Θᵣ[(k-1)*2ny+ny+(1:ny),:] = Θ[idx,:]
  end
  for k = 1:nb
    idx = (k-1)*2ny+(1:ny) + ny
    Θᵣ[(k-1)*2ny+2ny+(1:ny),:] = Θ[idx,:]
  end

  # /TODO simulation and prediction for MIMO ARX
  return Θᵣ, mse
end

function _mse{T}(Θ::AbstractMatrix{T}, Φ::AbstractMatrix{T},
  Y::AbstractMatrix{T})
  return reshape(sumabs2(Φ*Θ-Y,1)/size(Y,1),size(Y,2))
end

# Auxilliary methods

function _ls_toeplitz{M1<:BlockToeplitz,M2<:AbstractArray}(Φ::M1, Y::M2)
  Q,R = qr(full(Φ))
  return R\(Q.'*Y)
#  return lstoeplitz(Φ, Y)[1]
#  Q,R =  qrtoeplitz(Φ)
#  return R\(Q.'*Y)
end

function _fill_truncate_ic{T<:Real,M1<:AbstractArray,M2<:AbstractArray}(
  data::IdDataObject{T,M1,M2}, md::Int, m::Int; feedthrough::Bool=false)
  y,u = data.y, data.u
  ny  = data.ny
  nu  = data.nu

  l   = nu+ny

  Y   = feedthrough ? y[md+1:end,:] - u[md+1:end,:] :
                      y[md+1:end,:]
  # trim away delay in u and corresponding elements in y
  ywd = y[md-m+1:end-1,:]
  uwd = u[md-m+1:end-1,:]

  # fill row and col
  col = hcat(-ywd[m:end,:], uwd[m:end,:])
  row = zeros(T,1,l*m)
  for i = 1:m
    idx = m+1-i
    row[1,(i-1)*l+1:i*l-nu] = -ywd[idx,:]
    row[1,(i-1)*l+ny+1:i*l] =  uwd[idx,:]
  end

  _normalize_col_row_Y!(col,row,Y)
  Φ = Toeplitz(col,row)
  return Φ, Y
end

function _fill_zero_ic{T<:Real,M1<:AbstractArray,M2<:AbstractArray}(
  data::IdDataObject{T,M1,M2}, md::Int, m::Int; feedthrough::Bool=false)
  y,u = data.y, data.u
  ny  = data.ny
  nu  = data.nu
  l   = nu+ny

  Y   = feedthrough ? y[2:end,:] - u[2:end,:] :
                      y[2:end,:]
  ywd = y[1:end-1,:]
  uwd = u[1:end-1,:]

  # fill col and row
  col = hcat(-ywd, uwd)
  row = hcat(col[1:1,1:l], zeros(T,1,(m-1)*l))

  _normalize_col_row_Y!(col,row,Y)
  Φ = Toeplitz(col,row)
  return Φ, Y
end

function _normalize_col_row_Y!{M1<:AbstractMatrix,M2<:AbstractMatrix}(
  col::M1, row::M1, Y::M2)
  γ = norm(col)
  row /= γ
  col /= γ  # 5γ
  Y   /= γ
end
