function morsm{T,A1,A2,S,ORD<:FullPolyOrder}(
    data::IdDataObject{T,A1,A2}, model::PolyModel{S,ORD,OE},
    options::IdOptions=IdOptions(estimate_initial=false))

  x,pr      = _morsm(data, model, options)
  mse       = _mse(data, model, x, options)
  modelfit  = _modelfit(mse, data.y)
  idinfo    = OneStepIdInfo(mse, modelfit, model)
  a,b,f,c,d = _getpolys(model, x)

  IdMFD(a, b, f, c, d, data.Ts, idinfo)
end

function _morsm{T<:Real,A1,A2,S,ORD<:FullPolyOrder}(
    data::IdDataObject{T,A1,A2}, model::PolyModel{S,ORD,OE},
    options::IdOptions=IdOptions(estimate_initial=false))

  estimate_initial = options.estimate_initial

  y,u,N     = data.y,data.u,data.N
  ny,nu     = data.ny, data.nu
  ny == 1 || error("morsm: ny == 1 only supported")
  na,nb,nf,nc,nd,nk = orders(model)

  # High order models to test filter
  minorder  = min(max(maximum(nb), maximum(nf)), Int(floor(N/2)))
  maxorder  = min(10*max(maximum(nb), maximum(nf)), Int(floor(N/2)))
  orderh    = min(maxorder+10, Int(floor(N/2)))
  nbrorders = min(10, maxorder-minorder+1)
  ordervec  = convert(Array{Int},round(linspace(minorder, maxorder, nbrorders)))

  # Model for cost function evaluation using orderh noise model
  if 1 == 1 # only G implemented first
    bjmodel = BJ(nb,nf,0,orderh,nk,ny,nu)
  else # version == :H
    bjmodel = BJ(nb,nf,nc,nd,nk,ny,nu)
  end

  # OE model for STMCB
  Gmodel    = OE(nb,nf,nk)

  # High-order model used for high order noise model
  modelₕ = ARX(orderh, orderh, 1)
  Θₕ, peharx = _arx(data, modelₕ, options)
  mₕ  = ny*orderh+nu*orderh
  xr  = reshape(Θₕ[1:mₕ*ny], mₕ, ny)
  xaₕ = view(xr, 1:ny*orderh, :)
  Aₕ  = PolyMatrix(vcat(eye(T,ny),      _blocktranspose(xaₕ, ny, ny, orderh)), (ny,ny))

  # filtered data
  yf = similar(y)
  uf = similar(u)
  dataf = iddata(yf, uf, data.Ts)

  mₗ = nu*ny*(nb+nf)
  bestx   = zeros(T,mₗ+ny^2*na)
  bestpe  = typemax(Float64)
  for m in ordervec
    modelₗ = ARX(m, m, 1)
    Θₗ  = _arx(data, modelₗ, options)[1]
    xr  = reshape(Θₗ, ny*m+nu*m, ny) # [1:(ny*m+nu*m)*ny]
    xaₗ = view(xr, 1:ny*m, :)
    xbₗ = view(xr, ny*m+(1:nu*m), :)
    Aₗ  = PolyMatrix(vcat(eye(T,ny),      _blocktranspose(xaₗ, ny, ny, m)), (ny,ny))
    Bₗ  = PolyMatrix(vcat(zeros(T,ny,nu), _blocktranspose(xbₗ, ny, nu, m)), (ny,nu))

    _filt_fir!(uf, Aₗ, u)
    if 1 == 1
      _filt_fir!(yf, Bₗ, u)
    else # filter == :data
      _filt_fir!(yf, Aₗ, y)
    end
    #dataf = iddata(yf, uf, data.Ts)
    ΘG    = _stmcb(dataf, Gmodel, options)

    if 1 == 1
      xg = reshape(ΘG, mₗ, ny)
      x  = vcat(xg, xaₕ) |> vec
      pe = cost(data, bjmodel, x, options)
    else # version == :H
      # create noise estimate
      # yef    = filt(Aₗ, Iₗ, yf) - filt(b,1,u) # vhat
      # uef    = filt(Aₗ, Iₗ,yef)             # ehat = Hhat^-1 vhat
      # dataef = iddata(yef, uef, data.Ts)
      # stmcbnoise = STMCB(ic, true, maxiter, tol)
      # ΘH,~   = _stmcb(dataef, [nc; nd; nk], stmcbnoise)
      # x      = vcat(ΘG[1:nb], ΘH[1:nc], ΘH[nc+1:end], ΘG[nb+1:end])
      # pe     = calc_bj(data, n, x, BJ(ic=ic))
    end

    if pe < bestpe
      bestpe = pe
      bestx  = x
    end
  end
  return bestx, bestpe
end

# Matlab type of model structure
function _morsm{T<:Real,A1,A2,S}(
  data::IdDataObject{T,A1,A2}, model::PolyModel{S,MPolyOrder,OE},
  options::IdOptions=IdOptions(estimate_initial=false))
  any(orders(model)[1] .> 0) && error("MORSM: A can not be estimated")

  na,nb,nf,nc,nd,nk = orders(model)
  ny,nu             = data.ny,data.nu
  y,u               = data.y,data.u
  b  = Vector{Vector{T}}(ny)
  f  = Vector{Vector{T}}(ny)
  pe = Vector{T}(ny)
  for i = 1:ny
    datai     = iddata(view(y,i:i,:), u, data.Ts) #
    morderi   = MPolyOrder(na[i:i,:], nb[i:i,:], nf[i:i,:], na[i:i,:], na[i:i,:], nk[i:i,:])
    modeli    = PolyModel(morderi, ny, nu, Val{:mimo}, OE)
    bᵢ,fᵢ,peᵢ = _morsm_yi(datai, modeli, options)
    b[i]      = bᵢ
    f[i]      = fᵢ
    pe[i]     = peᵢ
  end

  return vcat(vcat(b...), vcat(f...)), pe
end

function _morsm_yi{T<:Real,A1,A2,S,U}(
  data::IdDataObject{T,A1,A2}, model::PolyModel{S,MPolyOrder,U},
  options::IdOptions=IdOptions(estimate_initial=false))
  y,u,N     = data.y,data.u,data.N
  ny,nu     = 1, data.nu  # ny = 1

  na,nb,nf,nc,nd,nk = orders(model)

  # High order models to test filter
  minorder  = min(max(maximum(nb), maximum(nf)), Int(floor(N/2)))
  maxorder  = min(10*max(maximum(nb), maximum(nf)), Int(floor(N/2)))
  orderh    = min(maxorder+10, Int(floor(N/2)))
  nbrorders = min(10, maxorder-minorder+1)
  ordervec  = convert(Array{Int},round(linspace(minorder, maxorder, nbrorders)))

  # High-order model used for high order noise model
  modelₕ     = ARX(orderh, orderh, ones(Int,nu), ny, nu)
  Θₕ, peharx = _arx(data, modelₕ, options)
  #mₕ  = orderh+nu*orderh
  xaₕ = view(Θₕ, 1:orderh, 1:1)
  Aₕ  = PolyMatrix(vcat(eye(T,ny), _blocktranspose(xaₕ, ny, ny, orderh)), (ny,ny))

  # filtered data
  yf = similar(u,1,N)
  uf = similar(u,1,N)

  mₗ      = nu*ny*(nb+nf)
  b       = Vector{Vector{T}}(length(nb))
  f       = Vector{Vector{T}}(length(nf))
  bestb   = Vector{Vector{T}}(length(nb))
  bestf   = Vector{Vector{T}}(length(nf))
  bestpe  = typemax(Float64)
  for m in ordervec
    modelₗ = ARX(m, m, ones(Int,nu), ny, nu)
    Θₗ  = _arx(data, modelₗ, options)[1]
    xr  = reshape(Θₗ, ny*m+nu*m, ny) # [1:(ny*m+nu*m)*ny]
    xaₗ = view(xr, 1:ny*m, :)
    xbₗ = view(xr, ny*m+(1:nu*m), :)
    Aₗ  = PolyMatrix(vcat(eye(T,ny),      _blocktranspose(xaₗ, ny, ny, m)), (ny,ny))
    Bₗ  = PolyMatrix(vcat(zeros(T,ny,nu), _blocktranspose(xbₗ, ny, nu, m)), (ny,nu))

    yp = zeros(T, size(y))
    for iu in 1:nu
      _filt_fir!(uf, Aₗ, view(u,iu:iu,:))
      _filt_fir!(yf, Bₗ[1:1,iu:iu], view(u,iu:iu,:))
      nbi,nfi,nki = nb[1,iu],nf[1,iu],nk[1,iu]
      dataf   = iddata(yf, uf, data.Ts)

      # OE model for STMCB
      Gmodel  = OE(nbi,nfi,1,1,1)

      ΘG = _stmcb(dataf, Gmodel, options)
      xg = reshape(ΘG, nbi+nfi, ny)
      b[iu] = xg[1:nbi]
      f[iu] = xg[nbi+(1:nfi)]

      tmp = filt(vcat(zeros(T,nki),b[iu]), vcat(ones(T,1),f[iu]), view(u,iu,:))
      yp[:] += filt(vcat(zeros(T,nki),b[iu]), vcat(ones(T,1),f[iu]), view(u,iu,:))
    end
    pe = cost(y, yp, N, options)

    if pe < bestpe
      bestb = b
      bestf = f
      bestpe = pe
    end
  end
  return vcat(bestb...), vcat(bestf...), bestpe
end
