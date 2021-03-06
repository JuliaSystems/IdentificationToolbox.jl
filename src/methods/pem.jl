function g!{T1<:Real, S<:PolyModel, T2<:Real, O}(
    data::IdDataObject{T1}, model::S, x::Vector{T2},
    last_x::Vector{T2}, last_V::Vector{T2}, g, storage::Matrix{T2},
    options::IdOptions{O}=IdOptions())
  gradhessian!(data, model, x, last_x, last_V, storage, options)
  copy!(g, storage[:, end])
end

function h!{T1<:Real, S<:PolyModel, T2<:Real, O}(
    data::IdDataObject{T1}, model::S, x::Vector{T2},
    last_x::Vector{T2}, last_V::Vector{T2}, H, storage::Matrix{T2},
    options::IdOptions{O}=IdOptions())
  gradhessian!(data, model, x, last_x, last_V, storage, options)
  copy!(H, storage[:,1:end-1])
end

# function pem{M1<:OneStepIdMethod, T1<:Real}(
#     data::IdDataObject{T1}, n::Vector{Int}, method::M1)
#   @assert data.ny < 2 string("PEM only implemented for SISO systems")
#   @assert data.nu < 2 string("PEM only implemented for SISO systems")
#
#   fitmodel(data, n, method)
# end
#
# function pem{M1<:IterativeIdMethod, M2<:OneStepIdMethod, T1<:Real}(
#     data::IdDataObject{T1}, n::Vector{Int}, method::M1,
#     init_method::M2=MORMSM(version=:H), n_init::Vector{Int}=n)
#   @assert data.ny < 2 string("PEM only implemented for SISO systems")
#   @assert data.nu < 2 string("PEM only implemented for SISO systems")
#
#   s1 = fitmodel(data, n_init, init_method)
#   x0  = get_param(s1,n)
#   pem(data, n, x0, method)
# end
function pem{T1<:Real,V1,V2}(
  data::IdDataObject{T1,V1,V2}, sysinit::IdMFD, options::IdOptions=IdOptions())
  pem(data, sysinit.info.model, get_params(sysinit), options)
end

function pem{S<:PolyModel, T1<:Real, T2<:Real,V1,V2}(
    data::IdDataObject{T1,V1,V2}, model::S, x0::AbstractVector{T2}, options::IdOptions=IdOptions())

  k = length(x0) # number of parameters
  last_x  = ones(T2,k)
  last_V  = -ones(T2,1)


  opt::Optim.OptimizationResults
  if !options.OptimizationOptions.autodiff
    storage = zeros(k, k+1)
    df = TwiceDifferentiable(x -> cost(data, model, x, options),
    (x,g) -> g!(data, model, x, last_x, last_V, g, storage, options),
    (x,H) -> h!(data, model, x, last_x, last_V, H, storage, options))
    opt = optimize(df, x0, Newton(), options.OptimizationOptions)
  else
    opt = optimize(x->cost(data, model, x, options),
          x0, Newton(), options.OptimizationOptions)
  end

  mse       = _mse(data, model, opt.minimizer, options)
  modelfit  = _modelfit(mse, data.y)
  idinfo    = IterativeIdInfo(mse, modelfit, opt, model)
  Θₚ,icbf,icdc,iccda = _split_params(model, opt.minimizer, options)
  A,B,F,C,D = _getpolys(model, Θₚ)
  _getmodel(A,B,F,C,D,data.Ts,idinfo,model)
  #IdMFD(A,B,F,C,D,data.Ts,idinfo)
end

# methods needed until ControlCore is updated with proper TransferFunction Type
# Mpoly models just get the corresponding polynomial matrices back
_getmodel{S,M}(A,B,F,C,D,Ts,idinfo,Model::PolyModel{S,FullPolyOrder{S},M}) =
  IdMFD(A,B,F,C,D,Ts,idinfo)
_getmodel(A,B,F,C,D,Ts,idinfo,Model) = A,B,F,C,D,idinfo

function _blocktranspose{T<:Real}(x::AbstractMatrix{T}, ny::Int, nu::Int, nx::Int)
  nx == 0 && return zeros(T,0,nu)
  r = zeros(T, nx*ny, nu)
  for ix = 0:nx-1
    r[ix*ny+(1:ny),:] = x[ix*nu+(1:nu),:].'
  end
  return r
end

function _getpolys{T<:Real,S,M}(model::PolyModel{S,
    FullPolyOrder{S},M}, x::Vector{T})
  na,nb,nf,nc,nd,nk = orders(model)
  ny,nu = model.ny, model.nu

  m  = ny*(na+nf+nc+nd)+nu*nb
  xr = reshape(x[1:m*ny], m, ny)

  xa = _blocktranspose(view(xr,                       1:ny*na, :), ny, ny, na)
  xb = _blocktranspose(view(xr, ny*na+              (1:nu*nb), :), ny, nu, nb)
  xf = _blocktranspose(view(xr, ny*na+nu*nb+        (1:ny*nf), :), ny, ny, nf)
  xc = _blocktranspose(view(xr, ny*(na+nf)+nu*nb+   (1:ny*nc), :), ny, ny, nc)
  xd = _blocktranspose(view(xr, ny*(na+nf+nc)+nu*nb+(1:ny*nd), :), ny, ny, nd)

  # zero pad vectors
  A = PolyMatrix(vcat(eye(T,ny),         xa), (ny,ny), :z̄)
  B = PolyMatrix(vcat(zeros(T,ny*nk[1],nu), xb), (ny,nu), :z̄) # TODO fix nk
  F = PolyMatrix(vcat(eye(T,ny),         xf), (ny,ny), :z̄)
  C = PolyMatrix(vcat(eye(T,ny),         xc), (ny,ny), :z̄)
  D = PolyMatrix(vcat(eye(T,ny),         xd), (ny,ny), :z̄)

  return A,B,F,C,D
end

function _mse{T<:Real,S<:PolyModel, O}(data::IdDataObject{T}, model::S, x, options::IdOptions{O}=IdOptions())
  y,N     = data.y,data.N
  y_est = predict(data, model, x, options)
  sumabs2(y-y_est,2)[:]/N
end

function _modelfit{T<:Real}(mse, y::AbstractVector{T})
  ny = size(y,2)
  modelfit = 100*(1 - mse/cov(y)) # TODO fix to correct order m y[m:N]
end

function _modelfit{T<:Real}(mse, y::AbstractMatrix{T})
  ny,N = size(y)
  modelfit = [100*(1 - mse[i]/cov(y[i,1:N])) for i in 1:ny] # TODO fix to correct order m y[m:N]
end

predict{T1,V1,V2}(data::IdDataObject{T1,V1,V2}, sys::IdMFD) =
  _predict(data,sys.A,sys.B,sys.F,sys.C,sys.D,sys.info.model)


function predict{T1,V1,V2,S,U,M,T2,O}(data::IdDataObject{T1,V1,V2},
  model::PolyModel{S,U,M}, Θ::AbstractVector{T2}, options::IdOptions{O}=IdOptions())
  Θₚ,icbf,icdc,iccda = _split_params(model, Θ, options)
  a,b,f,c,d          = _getpolys(model, Θₚ)

  return _predict(data,a,b,f,c,d,model,icbf,icdc,iccda)
  na,nb,nf,nc,nd,nk  = orders(model)

  ny   = data.ny
  nbf  = max(nb, nf)
  ndc  = max(nd, nc)
  ncda = max(nc, nd+na)

  # save unnecessary computations
  temp  = nbf > 0 ? filt(b, f, data.u, icbf) : data.u
  temp2 = ndc > 0 ? filt(d, c, temp, icdc) : temp
  temp3 = ncda > 0 ? temp2 + filt(c-d*a, c, data.y, iccda) : temp2
  return temp3 # 10.53 [Ljung1999]
end

function _predict{T1,V1,V2,S,U,M}(data::IdDataObject{T1,V1,V2},a,b,f,c,d,
  model::PolyModel{S,U,M},icbf=zeros(T1,0,0),
  icdc=zeros(T1,0,0),iccda=zeros(T1,0,0))

  na,nb,nf,nc,nd,nk  = orders(model)

  ny   = data.ny
  nbf  = max(nb, nf)
  ndc  = max(nd, nc)
  ncda = max(nc, nd+na)

  if length(icbf) == 0
    icbf = zeros(T1, ny*nk[1], nbf)
  end
  if length(icdc) == 0
    icdc = zeros(T1, ny, ndc)
  end
  if length(iccda) == 0
    iccda = zeros(T1, ny, ncda)
  end

  # save unnecessary computations
  temp  = nbf > 0 ? filt(b, f, data.u, icbf) : data.u
  temp2 = ndc > 0 ? filt(d, c, temp, icdc) : temp
  temp3 = ncda > 0 ? temp2 + filt(c-d*a, c, data.y, iccda) : temp2
  return temp3 # 10.53 [Ljung1999]
end

function _split_params{S,U,M,O,T}(model::PolyModel{S,U,M}, Θ::AbstractArray{T}, options::IdOptions{O})
  na,nb,nf,nc,nd,nk = orders(model)

  ny,nu = model.ny,model.nu
  nbf   = max(nb, nf)
  ndc   = max(nd, nc)
  ncda  = max(nc, nd+na)
  m     = ny^2*(na+nf+nc+nd)+nu*ny*nb
  mi    = (ndc+nbf+ncda)*ny

  Θₚ = Θ[1:m]
  Θᵢ = options.estimate_initial ? Θ[m+1:m+mi] : zeros(T,mi)
  icbf  = nbf > 0  ? reshape(Θᵢ[1:nbf*ny], ny*nk[1], nbf)            : zeros(T,0,0)  # TODO fix nk
  icdc  = ndc > 0  ? reshape(Θᵢ[nbf*ny+(1:ndc*ny)], ny, ndc)         : zeros(T,0,0)
  iccda = ncda > 0 ? reshape(Θᵢ[(nbf+ndc)*ny+(1:ncda*ny)], ny, ncda) : zeros(T,0,0)
  return Θₚ, icbf, icdc, iccda
end

# calculate the value function V. Used for automatic differentiation
function cost{T<:Real,S,M<:AbstractModelOrder,P,O}(data::IdDataObject{T}, model::PolyModel{S,M,P}, x,
    options::IdOptions{O}=IdOptions())
  y,N   = data.y,data.N
  y_est = predict(data, model, x, options)
  return cost(y, y_est, N, options)
end

cost{T}(y::AbstractArray{T}, y_est, N::Int, options::IdOptions) =
  value(options.loss_function, y, y_est, AvgMode.Sum())/2N

function _getpolys{T<:Real,S,M}(model::PolyModel{S,
    MPolyOrder,M}, Θ::Vector{T})
  a,b,f,c,d = _getmatrix(model, Θ)
  A = map(x->Poly(x, :z̄),a)  |> PolyMatrix
  B = map(x->Poly(x, :z̄),b)  |> PolyMatrix
  F = map(x->Poly(x, :z̄),f)  |> PolyMatrix
  C = map(x->Poly(x, :z̄), c) |> diagm |> PolyMatrix
  D = map(x->Poly(x, :z̄), d) |> diagm |> PolyMatrix
  return A,B,F,C,D
end

function _getmatrix{T<:Real,S,M}(model::PolyModel{S,
    MPolyOrder,M}, Θ::Vector{T})
  na,nb,nf,nc,nd,nk = orders(model)
  ny,nu             = model.ny,model.nu
  Na,Nb,Nf,Nc,Nd    = sum(na),sum(nb),sum(nf),sum(nc),sum(nd)

  a = view(Θ,1:Na)
  b = view(Θ,Na+(1:Nb))
  f = view(Θ,Na+Nb+(1:Nf))
  c = view(Θ,Na+Nb+Nf+(1:Nc))
  d = view(Θ,Na+Nb+Nf+Nc+(1:Nd))

  A = Matrix{Vector{T}}(ny,ny)
  B = Matrix{Vector{T}}(ny,nu)
  F = Matrix{Vector{T}}(ny,nu)
  C = Vector{Vector{T}}(ny)
  D = Vector{Vector{T}}(ny)

  ma=mb=mf=mc=md=0
  for i = 1:ny
    for j = 1:ny
      if i == j
        A[i,j] = vcat(ones(T,1), a[ma+(1:na[i,j])])
        C[i]   = vcat(ones(T,1), a[ma+(1:na[i,j])])
        D[i]   = vcat(ones(T,1), a[ma+(1:na[i,j])])
        ma  += na[i,j]
        mc  += nc[i]
        md  += nd[i]
      else
        A[i,j] = vcat(zeros(T,1), a[ma+(1:na[i,j])])
        ma += na[i,j]
      end
    end
    for j = 1:nu
      B[i,j] = vcat(zeros(T,nk[i,j]), b[mb+(1:nb[i,j])])
      F[i,j] = vcat(ones(T,1), f[mf+(1:nf[i,j])])
      mb    += nb[i,j]
      mf    += nf[i,j]
    end
  end
  return A,B,F,C,D
end

function _split_params{S,M,O,T}(model::PolyModel{S,MPolyOrder,M}, Θ::AbstractArray{T}, options::IdOptions{O})
  na,nb,nf,nc,nd,nk = orders(model)

  icbf  = zeros(T,0,0)
  icdc  = zeros(T,0,0)
  iccda = zeros(T,0,0)
  return Θ, icbf, icdc, iccda
end

function predict{T1,A1,A2,S,P,T2,O}(data::IdDataObject{T1,A1,A2},
    model::PolyModel{S,MPolyOrder,P}, Θ::AbstractVector{T2}, options::IdOptions{O}=IdOptions())

  na,nb,nf,nc,nd,nk  = orders(model)
  N,ny,nu            = data.N,data.ny,data.nu
  Na,Nb,Nf,Nc,Nd     = sum(na),sum(nb),sum(nf),sum(nc),sum(nd)
  a,b,f,c,d          = _getmatrix(model, Θ)
  T = promote_type(T1,T2)

  out = zeros(T, ny, N)
  for i = 1:ny
    aᵢ = view(a,i,:)
    bᵢ = view(b,i,:)
    fᵢ = view(f,i,:)
    cᵢ = c[i]
    dᵢ = d[i]
    _predict_i!(view(out,i,:),data,model,i,aᵢ,bᵢ,fᵢ,cᵢ,dᵢ,
      view(na,i,:), view(nb,i,:), view(nf,i,:), view(nk,i,:))
  end
  return out
end

function _predict_i!{T1,T2,A1,A2,S,P}(out,data::IdDataObject{T1,A1,A2},
    model::PolyModel{S,MPolyOrder,P}, i::Int, a::AbstractArray{Vector{T2}},
    b::AbstractArray{Vector{T2}}, f::AbstractArray{Vector{T2}}, c::Vector{T2},
    d::Vector{T2}, na::AbstractArray{Int}, nb::AbstractArray{Int},
    nf::AbstractArray{Int}, nk::AbstractArray{Int})
  y,u     = data.y,data.u
  ny,nu,N = data.ny,data.nu,data.N
  for j in 1:nu
    num = _poly_mul(a[i], _poly_mul(d, b[j]))
    den = _poly_mul(c, f[j])
    out[:] += filt(num, den, view(u,j,:))
  end
  for j in 1:ny
    if j == i
      continue
    end
    num = _poly_mul(a[i], _poly_mul(d, a[j]))
    out[:] += filt(num, c, view(y,j,:))
  end

  tmp = _poly_mul(d,a[i])
  num = vcat(c, zeros(T2, length(tmp)-length(c))) - tmp
  out[:] += filt(num, c, view(y,i,:))
end

function _poly_mul(a, b)
  T = promote_type(eltype(a), eltype(b))
  n = length(a)-1
  m = length(b)-1
  r = zeros(T,m+n+1)
  @inbounds for i in eachindex(a)
    for j in eachindex(b)
      r[i+j-1] += a[i] * b[j]
    end
  end
  return r
end
