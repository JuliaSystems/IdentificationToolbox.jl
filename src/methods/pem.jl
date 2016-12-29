function cost{T<:Real,S<:PolynomialModel, O}(
  data::IdDataObject{T}, model::S, x, last_x, last_V, storage,
  options::IdOptions{O}=IdOptions())
  cost(data, model, x, options)
end

function g!{T1<:Real, S<:PolynomialModel, T2<:Real, O}(
    data::IdDataObject{T1}, model::S, x::Vector{T2},
    last_x::Vector{T2}, last_V::Vector{T2}, g, storage::Matrix{T2},
    options::IdOptions{O}=IdOptions())
  gradhessian!(data, model, x, last_x, last_V, storage, options)
  copy!(g, storage[:, end])
end

function h!{T1<:Real, S<:PolynomialModel, T2<:Real, O}(
    data::IdDataObject{T1}, model::S, x::Vector{T2},
    last_x::Vector{T2}, last_V::Vector{T2}, H, storage::Matrix{T2},
    options::IdOptions{O}=IdOptions())
  gradhessian!(data, model, x, last_x, last_V, storage, options)
  copy!(H, storage[:,1:end-1])
end

function pem{M1<:OneStepIdMethod, T1<:Real}(
    data::IdDataObject{T1}, n::Vector{Int}, method::M1)
  @assert data.ny < 2 string("PEM only implemented for SISO systems")
  @assert data.nu < 2 string("PEM only implemented for SISO systems")

  fitmodel(data, n, method)
end

function pem{M1<:IterativeIdMethod, M2<:OneStepIdMethod, T1<:Real}(
    data::IdDataObject{T1}, n::Vector{Int}, method::M1,
    init_method::M2=MORMSM(version=:H), n_init::Vector{Int}=n)
  @assert data.ny < 2 string("PEM only implemented for SISO systems")
  @assert data.nu < 2 string("PEM only implemented for SISO systems")

  s1 = fitmodel(data, n_init, init_method)
  x0  = get_param(s1,n)
  pem(data, n, x0, method)
end

function pem{S<:PolynomialModel, T1<:Real, T2<:Real}(
    data::IdDataObject{T1}, model::S, x0::AbstractVector{T2}; options::IdOptions=IdOptions())

  k = length(x0) # number of parameters
  last_x  = ones(T2,k)
  last_V  = -ones(T2,1)


  opt::Optim.OptimizationResults
  if !options.OptimizationOptions.autodiff
    storage = zeros(k, k+1)
    df = TwiceDifferentiableFunction(x    -> cost(data, model, x, options),
    (x,g) -> g!(data, model, x, last_x, last_V, g, storage, options),
    (x,H) -> h!(data, model, x, last_x, last_V, H, storage, options))
    opt = optimize(df, x0, Newton(), options.OptimizationOptions)
  else
    opt = optimize(x->cost(data, model, x, options),
          x0, Newton(), options.OptimizationOptions)
  end

  println(fieldnames(opt))
  mse       = _mse(data, model, opt.minimizer)
  modelfit  = _modelfit(mse, data.y)
  println(mse)
  println(modelfit)
  idinfo    = IterativeIdInfo(mse, modelfit, opt, model)
  Θₚ,icbf,icdc,iccda = _split_params(model, opt.minimizer, IdOptions())
  A,B,F,C,D = _getpolys(model, Θₚ)

  IdMFD(A,B,F,C,D,data.Ts,idinfo)
end

function _getpolys{T<:Real,S}(model::PolynomialModel{FullPolynomialOrder{S}},
    x::Vector{T})
  na,nb,nf,nc,nd,nk = orders(model)

  naf = na + nf - 1
  nad = na + nd - 1

  mG = max(naf, nb)
  mH = max(nad, nc)

  a = vcat(ones(T,1),   x[             1:na ], zeros(T, mG-nb))
  b = vcat(zeros(T,nk), x[na+         (1:nb)], zeros(T, mH-nc))
  f = vcat(ones(T,1),   x[na+nb+      (1:nf)])
  c = vcat(ones(T,1),   x[na+nb+nf+   (1:nc)])
  d = vcat(ones(T,1),   x[na+nb+nf+nc+(1:nd)])


  A = Poly(reverse(a))
  B = Poly(reverse(b))
  F = Poly(reverse(f))
  C = Poly(reverse(c))
  D = Poly(reverse(d))

  return A,B,F,C,D
end

function _blocktranspose{T<:Real}(x::AbstractMatrix{T}, ny::Int, nu::Int, nx::Int)
  nx == 0 && return zeros(T,0,nu)
  r = zeros(T, nx*ny, nu)
  for ix = 0:nx-1
    r[ix*ny+(1:ny),:] = x[ix*nu+(1:nu),:].'
  end
  return r
end

function _getpolys{T<:Real,S}(model::PolynomialModel{FullPolynomialOrder{S}},
    x::Vector{T})
  na,nb,nf,nc,nd = orders(model)
  ny,nu = model.ny, model.nu

  naf = na + nf - 1
  nad = na + nd - 1

  mG = max(naf, nb)
  mH = max(nad, nc)

  m  = ny*(na+nf+nc+nd)+nu*nb
  xr = reshape(x, m, ny)

  xa = _blocktranspose(view(xr,                       1:ny*na, :), ny, ny, na)
  xb = _blocktranspose(view(xr, ny*na+              (1:nu*nb), :), ny, nu, nb)
  xf = _blocktranspose(view(xr, ny*na+nu*nb+        (1:ny*nf), :), ny, ny, nf)
  xc = _blocktranspose(view(xr, ny*(na+nf)+nu*nb+   (1:ny*nc), :), ny, ny, nc)
  xd = _blocktranspose(view(xr, ny*(na+nf+nc)+nu*nb+(1:ny*nd), :), ny, ny, nd)

  # zero pad vectors
  A = PolyMatrix(vcat(eye(T,ny),      xa), (ny,ny))
  B = PolyMatrix(vcat(zeros(T,ny,nu), xb), (ny,nu))
  F = PolyMatrix(vcat(eye(T,ny),      xf), (ny,ny))
  C = PolyMatrix(vcat(eye(T,ny),      xc), (ny,ny))
  D = PolyMatrix(vcat(eye(T,ny),      xd), (ny,ny))

  return A,B,F,C,D
end

function _mse{T<:Real,S<:PolynomialModel}(data::IdDataObject{T}, model::S, x)
  y     = data.y
  N     = size(y,1)
  y_est = predict(data, model, x)
  sumabs2(y-y_est,1)[:]/N
end

function _modelfit{T<:Real}(mse, y::AbstractVector{T})
  ny = size(y,2)
  modelfit = 100*(1 - mse/cov(y[1:N])) # TODO fix to correct order m y[m:N]
end

function _modelfit{T<:Real}(mse, y::AbstractMatrix{T})
  ny = size(y,2)
  modelfit = [100*(1 - mse[i]/cov(y[1:N,i])) for i in 1:ny] # TODO fix to correct order m y[m:N]
end

# function predict{T1<:Real,V1,V2,T2<:Real}(
#     data::IdDataObject{T1,V1,V2},
#     model::PolynomialModel, Θ::Vector{T2}; ic::Symbol=:zero)
#   na,nb,nf,nc,nd,nk = orders(model.orders)
#   y,u      = data.y, data.u
#   ny,nu    = data.ny, data.nu
#
#   a,b,f,c,d = _getpolys(model, Θ)
#   # 10.53 [Ljung1999]
#   return filt(d,c,filt(b, f, u)) + filt(c-d*a,c,y)
#   #est(y,u,b,f,nk,ic)
# end

function predict{T1,V1,V2,S,M,O}(data::IdDataObject{T1,V1,V2},
  model::PolynomialModel{S,M}, Θ, options::IdOptions{O}=IdOptions())
  Θₚ,icbf,icdc,iccda = _split_params(model, Θ, options)
  a,b,f,c,d          = _getpolys(model, Θₚ)
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

function _split_params{S,M,O}(model::PolynomialModel{S,M}, Θ, options::IdOptions{O})
  na,nb,nf,nc,nd,nk = orders(model)

  ny   = model.ny
  nbf  = max(nb, nf)
  ndc  = max(nd, nc)
  ncda = max(nc, nd+na)
  m  = ny^2*(na+nf+nc*nd)+nu*ny*nb
  mi = (ndc+nbf+ncda)*ny

  Θₚ = Θ[1:m]
  Θᵢ = options.estimate_initial ? Θ[m+1:m+mi] : zeros(mi)
  icbf  = nbf > 0  ? reshape(Θᵢ[1:nbf*ny], nbf, ny)                  : zeros(0,0)
  icdc  = ndc > 0  ? reshape(Θᵢ[nbf*ny+(1:ndc*ny)], nbf, ny)         : zeros(0,0)
  iccda = ncda > 0 ? reshape(Θᵢ[(nbf+ndc)*ny+(1:ncda*ny)], nbf, ny) : zeros(0,0)
  return Θₚ, icbf, icdc, iccda
end

# calculate the value function V. Used for automatic differentiation
function cost{T<:Real,S<:PolynomialModel,O}(data::IdDataObject{T}, model::S, x,
    options::IdOptions{O}=IdOptions())
  y     = data.y
  N     = size(y,1)
  y_est = predict(data, model, x, options)

  # if ic == :truncate
  #   #TODO proper m
  #   m = 0
  #   return sumabs2(y-y_est)/(N-m)
  # end
  return cost(y, y_est, N, options)
end

cost{T}(y::AbstractArray{T}, y_est, N::Int, options::IdOptions) =
  sumvalue(options.loss_function, y, y_est)/N
#sumabs2(y-y_est)/N

# Returns the value function V and saves the gradient/hessian in `storage` as storage = [H g].
function gradhessian!{T<:Real, T2<:Real, S<:PolynomialModel, O}(
    data::IdDataObject{T}, model::S, x::Vector{T2},
    last_x::Vector{T2}, last_V::Vector{T2}, storage::Matrix{T2}, options::IdOptions{O}=IdOptions())
  # check if this is a new point
  if x != last_x
    # update last_x
    copy!(last_x, x)

    y  = data.y
    ny = data.ny
    N  = size(y,1)

    Psit  = psit(data, model, x)  # Psi the same for all outputs
    y_est = predict(data, model, x, options)
    eps   = y - y_est
    V     = cost(y, y_est, N, options)

    k = size(Psit,2)

    gt = zeros(T,1,k)
    H  = zeros(T,k,k)

    A_mul_B!(H,  Psit.', Psit)          # H = Psi*Psi.'
    for i = 0:ny-1
      A_mul_B!(gt, -eps[:,i+1].', Psit)   # g = -Psi*eps
      storage[i*k+(1:k), ny*k+1]    = 2*gt.'/N
      storage[i*k+(1:k), i*k+(1:k)] = H/N
    end

    # normalize
#    storage[1:k, k+1] /= N-m+1
#    storage[1:k, 1:k] /= N-m+1

    # update last_V
    copy!(last_V, V)

    return V
  end
  return last_V[1]
end
