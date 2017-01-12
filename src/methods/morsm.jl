function _morsm{T<:Real,A1,A2,S,U}(
  data::IdDataObject{T,A1,A2}, model::PolyModel{S,U,OE},
  options::IdOptions=IdOptions(estimate_initial=false))
  estimate_initial = options.estimate_initial

  y,u,N     = data.y,data.u,data.N
  ny,nu     = data.ny, data.nu
  ny == 1 || error("morsm: ny == 1 only supported")
  na,nb,bf,nc,nd,nk = orders(model)

  # High order models to test filter
  minorder  = convert(Int,max(floor(N/1000),2*(nb+nf)))
  maxorder  = convert(Int,min(floor(N/20),40))
  orderh    = maxorder+10
  nbrorders = min(10, maxorder-minorder)
  ordervec  = convert(Array{Int},round(linspace(minorder, maxorder, nbrorders)))
  println(ordervec)

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



# immutable MORSM <: OneStepIdMethod
#   ic::Symbol
#   filter::Symbol
#   version::Symbol
#   loop::Symbol
#   nbrorders::Int
#   maxiter::Int
#   tol::Float64
#
#   @compat function (::Type{MORSM})(ic::Symbol=:truncate, filter::Symbol=:input,
#       version::Symbol=:G, loop::Symbol=:open, nbrorders::Int=5, maxiter::Int=5,
#       tol::Float64=1e-8)
#     @assert in(ic, Set([:truncate,:zero]))  string("ic must be either :truncate or :zero")
#     @assert in(version, Set([:G,:H]))       string("version must be either :G or :H")
#     @assert in(loop, Set([:open, :closed])) string("loop must be either :open or :closed")
#     @assert nbrorders > 0                   string("nbrorders need to be greater than zero")
#     @assert in(filter, Set([:input,:data])) string("filter must be either :input or :data")
#     @assert maxiter > 0                     string("maxiter need to be greater than zero")
#     @assert tol >= 0                        string("tol need to be greater or equal to zero")
#
#     new(ic, filter, version, loop, nbrorders, maxiter, tol)
#   end
# end
#
# function MORSM(;ic::Symbol=:truncate, filter::Symbol=:input,
#     version::Symbol=:G, loop::Symbol=:open, nbrorders::Int=50, maxiter::Int=10,
#     tol::Float64=1e-12)
#   MORSM(ic, filter, version, loop, nbrorders, maxiter, tol)
# end
#
# function fitmodel{T<:Real}(data::IdDataObject{T}, n::Vector{Int}, method::MORSM; kwargs...)
#   morsm(data, n, method)
# end
#
# function morsm(data::IdDataObject, n::Vector{Int}, method::MORSM=MORSM())
#   x, mse = _morsm(data, n, method)
#
#   if method.version == :G
#     nb,nf,nk = n
#     m        = max(nb, nf)+1
#   else # method.version == :H
#     nb,nc,nd,nf,nk = n
#     nfc            = nc + nf
#     nfd            = nd + nf
#     ndb            = nd + nk + nb - 1
#     m              = max(nc, nfc, nfd, ndb)+1
#   end
#   N        = size(data.y,1)
#   modelfit = 100 * (1 - sqrt((N-m)*mse) / norm(data.y[m:N]-mean(data.y[m:N])))
#
#   a,b,c,d,f = _getvec(n, x, method)
#   info      = OneStepIdInfo(mse, modelfit, method, n)
#   IdDSisoRational(a, b, c, d, f, data.Ts, info)
# end
#
# function _getvec{T<:Real}(n::AbstractVector{Int}, x::AbstractVector{T}, method::MORSM)
#   if method.version == :G
#     nb, nf, nk = n
#     nd = length(x-nb-nf)
#     gm = max(nb+nk-1,nf)
#     b = vcat(zeros(T,nk), x[1:nb])
#     f = vcat(ones(T,1), x[nb+1:nb+nf])
#     c = ones(T,1)
#     d = vcat(ones(T,1), x[nb+nf+1:end])
#   else # method.version == :H
#     nb, nc, nd, nf, nk = n
#     gm = max(nb+nk,nf)
#     hm = max(nc,nd)
#     b = vcat(zeros(T,nk), x[1:nb])
#     c = vcat(ones(T,1), x[nb+1:nb+nc])
#     d = vcat(ones(T,1), x[nb+nc+1:nb+nc+nd])
#     f = vcat(ones(T,1), x[nb+nc+nd+1:end])
#   end
#   a = ones(T,1)
#   return a,b,c,d,f
# end
#
# function _morsm{T<:Real,A1,A2}(
#   data::IdDataObject{T,A1,A2}, model::PolyModel{S,U,OE},
#   options::IdOptions=IdOptions(estimate_initial=false))
#   estimate_initial = options.estimate_initial
#
#   # filter    = method.filter
#   # version   = method.version
#   # loop      = method.loop
#   # nbrorders = method.nbrorders
#   # maxiter   = method.maxiter
#   # tol       = method.tol
#   y, u      = data.y, data.u
#   N         = size(y,1)
#   maxorder  = convert(Int,min(floor(N/20),40))
#   orderh    = maxorder+2
#   if method.version == :G
#     nb, nf, nk = n
#     nc         = 0
#     nd         = orderh
#   else #method.version == :H
#     nb,nc,nd,nf,nk = n
#   end
#   na,nb,bf,nc,nd,nk = orders(model)
#   Gmodel    = OE(nb,nf,nk)
#
#   n         = vcat(nb, nc, nd, nf, nk)
#   m         = nb+nf
#   minorder  = convert(Int,max(floor(N/1000),2*(nb+nf)))
#   orders    = convert(Array{Int},round(linspace(minorder, maxorder, nbrorders)))
#   @assert nb >= 0 && nf >= 0 string("nb and nf must be larger or equal to zero")
#   @assert nk >= 0            string("nk must be greater or equal to zero")
#
# # find high order noise model
#   # Θ       = _arx(data, [orderh; orderh; nk], ARX(ic,false))[1]
#   # ah      = Θ[1:orderh]
#   # bh      = Θ[orderh+1:end]
#   # Ah      = append!(ones(T,1), ah)
#   # Bh      = append!(zeros(T,nk), bh)
#
#   Sₕ = _arx(data, modelₕ, options)
#   Aₕ = Sₕ.A
#   Bₕ = Sₕ.B
#
#   Iₗ = PolyMatrix(eye(T,ny),(ny,ny))
#
#   bestx   = zeros(T,sum(n[1:4]))
#   bestpe  = typemax(Float64)
#   for m in orders
#     bjmodel = BJ(nb,nf,nc,nd,nk,ny,nu)
#
#     Sₗ = arx(data, modelₗ; options = options)
#     Aₗ = Sₗ.A
#     Bₗ = Sₗ.B
#
#     # Θ = _arx(data, [m; m; nk], ARX(ic,false))[1]
#     # a = append!(ones(T,1), Θ[1:m])
#     # b = append!(zeros(T,nk), Θ[m+1:end])
#     A = PolyMatrix(vcat(eye(T,ny),      xa), (ny,ny))
#     B = PolyMatrix(vcat(zeros(T,ny,nu), xb), (ny,nu))
#
#
#     uf = filt(Aₗ,1,u)
#     if filter == :input
#       yf = filt(Bₗ, Iₗ, u)
#     else # filter == :data
#       yf = filt(Aₗ, Iₗ, y)
#     end
#     dataf = iddata(yf, uf, data.Ts)
#     ΘG    = _stmcb(dataf, Gmodel, options)
#
#     if version == :G
#       pe = cost(data, bjmodel, x[:], options)
#       pe    = calc_bj(data, n, vcat(ΘG[1:nb], ah, ΘG[nb+1:end]) , BJ(ic=ic))
#       x     = vcat(ΘG[1:nb], ΘG[nb+1:end], ah)
#     else # version == :H
#       # create noise estimate
#       yef    = filt(a,1,y) - filt(b,1,u) # vhat
#       uef    = filt(a,1,yef)             # ehat = Hhat^-1 vhat
#       dataef = iddata(yef, uef, data.Ts)
#       stmcbnoise = STMCB(ic, true, maxiter, tol)
#       ΘH,~   = _stmcb(dataef, [nc; nd; nk], stmcbnoise)
#       x      = vcat(ΘG[1:nb], ΘH[1:nc], ΘH[nc+1:end], ΘG[nb+1:end])
#       pe     = calc_bj(data, n, x, BJ(ic=ic))
#     end
#
#     if pe < bestpe
#       bestpe = pe
#       bestx  = x
#     end
#   end
#   return bestx, bestpe
# end
#
# function morsmcl{V<:AbstractVector}(y::V, u::V, r::V, K, nl::Int, nf::Int, nc::Int,
#     nd::Int, nk::Int=1;
#     nbrorders::Int=50, filter::Symbol=:input,
#     ic::Symbol=:truncate, ts::Int=1, loop::Symbol=closed)
# @assert nl >= 0 && nf >= 0 string("nl and nf must be larger or equal to zero")
# @assert nk >= 0            string("nk must be greater or equal to zero")
# @assert loop == :open || loop == :closed
#   string("loop must be either :open or :closed")
# @assert nbrorders > 0      string("nbrorders need to be greater than zero")
# @assert filter == :input || version == :data
#   string("filter must be either :input or :data")
# @assert ic == :truncate || ic == :zero
#   string("ic must be either :truncate or :zero")
#
#   N        = length(y)
#   T        = eltype(y)
#   maxorder = convert(Int,min(floor(N/10),100))
#   minorder = convert(Int,max(floor(N/1000),2*(nl+nf)))
#   orders   = convert(Array{Int},round(linspace(minorder, maxorder, nbrorders)))
#
# # find high order noise model
#   orderh  = maxorder+2
#   Θ       = arx(y,u,orderh,orderh,nk,:zero)[1]
#   b       = [zeros(nk); Θ[1:orderh]]
#   a       = [1; Θ[orderh+1:end]]
#   Sh      = tf(ah,ah+bh,1)
#   Shr     = filt(numvec(Sh), denvec(Sh), r)
#
#   invHhat = tf(a,[1],1)
#   Hhat    = tf([1],a,1)
#   Heye    = tf([1],[1],1)
#   besta   = a
#   bestb   = b
#
#   bestpe  = typemax(Int)
#   bestG   = one(ControlCore.DSisoRational)
#   bestH   = one(ControlCore.DSisoRational)
#   uf      = zeros(T,N)
#   yf      = zeros(T,N)
#   uef     = zeros(T,N)
#   yef     = zeros(T,N)
#
#   uf = filt(ah,1,Shr)
#   yf = filt(bh,1,Shr)
#
#   # first estimate
#   Ghat2, pe  = stmcb(yf, uf, nb, nf, Heye, nk; maxiter=10, tol=1e-12)
#   Ghat, pe  = stmcb(yf, uf, nb, nf, Heye, nk; maxiter=10, tol=1e-12)
#
#   eh = filt(ah+bh,[1],y) - filt(bh,[1],r)
#   ev
#   uef = filt(ah,1,eh)
#
#   Hhat,~  = stmcb(eh, uef, nc, nd, Heye, 1; maxiter=3, tol=1e-12,feedthrough=true)
#
#   #Θ = arx(eh,uef,1,1,1,:zero;feedthrough=true)[1]
#
#   Hr = filt(denvec(Hhat), numvec(Hhat),r)
#   Hsr = filt([1],denvec(Ghat)+[0;numvec(Ghat)],Hr)
#
#   uf2 = filt(1,1,Hsr)
#   yf2 = filt([0;numvec(Ghat)],denvec(Ghat),Hsr)
#
#   Hsr = filt(ah,ah+bh,Hr)
#   uf2 = filt(1,1,Hsr)
#   yf2 = filt([0;numvec(Ghat)],denvec(Ghat),Hsr)
#
#
#   for m in orders
#     Θ = arx(y,u,m,m,nk,:zero)[1]
#     b = [zeros(T,nk); Θ[1:m]]
#     a = [1; Θ[m+1:end]]
#
#     uf = filt(a,1,u)
#     if filter == :input
#       yf = filt(b,1,u)
#     elseif filter == :data
#       yf = filt(a,1,y)
#     end
#     Ghat, pe  = stmcb(yf, uf, nl, nf, Heye, nk; maxiter=10, tol=1e-12)
#
#     if version == :H
#       yef = filt(a,1,y) - filt(b,1,u)
#       uef = filt(a,1,yef)
#       Hhat,~  = stmcb(yef, uef, nl, nf, Heye, 0; maxiter=10, tol=1e-12)
#       pe = predbj(Ghat,1/Hhat,u,y,m,nk)
#     elseif version == :G
#       pe = predbj(Ghat,invHhat,u,y,m,nk)
#     end
#
#     if pe < bestpe
#       bestpe = pe
#       bestG  = Ghat
#       bestH  = Hhat
#       besta  = a
#       bestb  = b
#     end
#   end
#   return bestG, bestH, bestb, besta, bestpe
# end
