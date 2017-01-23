
function arx{T<:Real}(data::IdDataObject{T}, n::Vector{Int},
  options::IdOptions=IdOptions(estimate_initial=false))
    arx(data, ARX(n[1],n[2],n[3]), options)
end

function arx{T<:Real, V1<:AbstractArray, V2<:AbstractArray}(
  data::IdDataObject{T,V1,V2}, na, nb, nk;
  options::IdOptions=IdOptions(estimate_initial=false))
    arx(data, ARX(na,nb,nk,data.ny,data.nu), options)
end

function arx{T,V1,V2,S,U}(
  data::IdDataObject{T,V1,V2}, model::PolyModel{S,U,ARX},
  options::IdOptions=IdOptions(estimate_initial=false))
  na,nb,nf,nc,nd,nk = orders(model)
  # TODO model order assertions
  y,u   = data.y, data.u
  ny,nu = data.ny,data.nu
  N     = size(y,1)

  # estimate model
  x, mse    = _arx(data, model, options)

#  mse       = _mse(data, model, x, options)
  modelfit  = _modelfit(mse, data.y)
  idinfo    = OneStepIdInfo(mse, modelfit, model)
  a,b,f,c,d = _getpolys(model, x)

  IdMFD(a, b, c, d, f, data.Ts, idinfo)
end

function predict{T1,V1,V2,V3<:AbstractVector,S,U}(
    data::IdDataObject{T1,V1,V2}, model::PolyModel{S,U,ARX}, x::V3,
    options::IdOptions=IdOptions(estimate_initial=false))
  return _predict_arx(data, model, x, options)
end

function _predict_arx{T1,V1,V2,T2,S,U}(data::IdDataObject{T1,V1,V2},
    model::PolyModel{S,U,ARX},Θ::AbstractVector{T2},
    options::IdOptions=IdOptions(estimate_initial=false))
  T = promote_type(T1, T2)
  m = max(na, nb+nk-1)
  estimate_initial = options.estimate_initial
  y,u,N = data.y,data.u,data.N

  a = append!(zeros(T,1),  Θ[1:na])
  b = append!(zeros(T,nk), Θ[na+1:na+nb])
  N = length(y)
  y_est = zeros(T,N)
  if estimate_initial
    # assume y_est = y for t=1:m to find initial states for the filters
    y_est[1:m]   = y[1:m]
    si           = filtic(b,  ones(T,1), filt(b,  ones(T,1), u[m:-1:1]), u[m:-1:1])
    si2          = filtic(-a, ones(T,1), filt(-a, ones(T,1), y[m:-1:1]), y[m:-1:1])
    y_est[m+1:N] = filt(b, ones(T,1), u[m+1:N], si) + filt(-a, ones(T,1), y[m+1:N], si2)
  else
    # zero initial conditions
    y_est = filt(b, ones(T,1), u) + filt(-a, ones(T,1), y)
  end
  return y_est
end

function _arx{T,A1,A2,S,U}(
    data::IdDataObject{T,A1,A2}, model::PolyModel{S,U,ARX},
    options::IdOptions=IdOptions(estimate_initial=false))
  na,nb,nf,nc,nd,nk = orders(model)
  y,u,N = data.y,data.u,data.N
  ny,nu = data.ny,data.nu
  estimate_initial = options.estimate_initial
  feedthrough = false # method.feedthrough
  @assert size(u,2)  == N "Input and output need have the same number of samples"
#  @assert length(nb) == nu "nb must have correct size"
#  @assert length(na) == ny "na must have correct size"
  @assert length(nk) == nu "nk must have correct size"
  @assert !feedthrough || ny == nu "Only square B matrices with feedthrough supported"
  @assert !any(nk .< 0) "Negative delays present"

  # handle variable nk
  if any(nk .== 0)
    ye = hcat(y[:,:], zeros(T,ny,1))
    ue = hcat(zeros(T,nu,1), u)
    nk += ones(nu)
  else
    ye = y[:,:]
    ue = u[:,:]
  end
  Nk = maximum(nk)
  ut = zeros(T,size(ue)...)
  for i = 1:nu
    if nk[i] > 1
      ut[i,nk[i]:end] = ue[i,1:end-nk[i]+1]
    else
      ut[i,:] = ue[i,:]
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

  if estimate_initial
    col,row,Y = _fill_truncate_ic(datat, md, m, feedthrough=feedthrough)
  else
    col,row,Y = _fill_zero_ic(datat, md, m, feedthrough=feedthrough)
  end
  _normalize_col_row_Y!(col,row,Y)
  Φ   = Toeplitz(col,row)
  Θ   = _ls_toeplitz(Φ, Y)
  mse = _mse(Θ, Φ, Y)

  # get Θ in the form Θ = [A₁ᵀ A₂ᵀ… B₁ᵀ B₂ᵀ …]ᵀ
  Θᵣ = zeros(T, nb*nu+na*ny,ny)
  for k = 1:na
    idx = (k-1)*2ny+(1:ny)
    Θᵣ[(k-1)*ny+(1:ny),:] = Θ[idx,:]
  end
  for k = 1:nb
    idx = (k-1)*(ny+nu)+(1:nu) + ny
    Θᵣ[na*ny+(k-1)*nu+(1:nu),:] = Θ[idx,:]
  end

  # /TODO simulation and prediction for MIMO ARX
  return vec(Θᵣ), mse
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
  y,u   = data.y.',data.u.'
  ny,nu = data.ny,data.nu
  l     = nu+ny

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

  return col,row,Y
end

function _fill_zero_ic{T<:Real,M1<:AbstractArray,M2<:AbstractArray}(
  data::IdDataObject{T,M1,M2}, md::Int, m::Int; feedthrough::Bool=false)
  y,u   = data.y.',data.u.'
  ny,nu = data.ny,data.nu
  l     = nu+ny

  Y   = feedthrough ? y[2:end,:] - u[2:end,:] :
                      y[2:end,:]
  ywd = y[1:end-1,:]
  uwd = u[1:end-1,:]

  # fill col and row
  col = hcat(-ywd, uwd)
  row = hcat(col[1:1,1:l], zeros(T,1,(m-1)*l))

  return col,row,Y
end

function _normalize_col_row_Y!{M1<:AbstractMatrix,M2<:AbstractMatrix}(
  col::M1, row::M1, Y::M2)
  γ = norm(col)
  row /= γ
  col /= γ  # 5γ
  Y   /= γ
end
