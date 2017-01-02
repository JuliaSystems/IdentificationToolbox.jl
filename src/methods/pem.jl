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

function pem{S<:PolyModel, T1<:Real, T2<:Real,V1,V2}(
    data::IdDataObject{T1,V1,V2}, model::S, x0::AbstractVector{T2}; options::IdOptions=IdOptions())

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

  mse       = _mse(data, model, opt.minimizer, options)
  modelfit  = _modelfit(mse, data.y)
  idinfo    = IterativeIdInfo(mse, modelfit, opt, model)
  Θₚ,icbf,icdc,iccda = _split_params(model, opt.minimizer, options)
  A,B,F,C,D = _getpolys(model, Θₚ)

  IdMFD(A,B,F,C,D,data.Ts,idinfo)
end

# function _getpolys{T<:Real,M}(model::PolyModel{ControlCore.Siso{true},
#     FullPolyOrder{ControlCore.Siso{true}},M}, x::Vector{T})
#   na,nb,nf,nc,nd,nk = orders(model)
#
#   a = vcat(ones(T,1),   x[             1:na ])
#   b = vcat(zeros(T,nk), x[na+         (1:nb)])
#   f = vcat(ones(T,1),   x[na+nb+      (1:nf)])
#   c = vcat(ones(T,1),   x[na+nb+nf+   (1:nc)])
#   d = vcat(ones(T,1),   x[na+nb+nf+nc+(1:nd)])
#
#
#   A = Poly(a) #Poly(reverse(a))
#   B = Poly(b) #Poly(reverse(b))
#   F = Poly(f) # Poly(reverse(f))
#   C = Poly(c) # Poly(reverse(c))
#   D = Poly(d) # Poly(reverse(d))
#
#   return A,B,F,C,D
# end

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
  na,nb,nf,nc,nd = orders(model)
  ny,nu = model.ny, model.nu

  m  = ny*(na+nf+nc+nd)+nu*nb
  xr = reshape(x[1:m*ny], m, ny)

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

function _mse{T<:Real,S<:PolyModel, O}(data::IdDataObject{T}, model::S, x, options::IdOptions{O}=IdOptions())
  y     = data.y
  N     = size(y,1)
  y_est = predict(data, model, x, options)
  sumabs2(y-y_est,1)[:]/N
end

function _modelfit{T<:Real}(mse, y::AbstractVector{T})
  ny = size(y,2)
  modelfit = 100*(1 - mse/cov(y)) # TODO fix to correct order m y[m:N]
end

function _modelfit{T<:Real}(mse, y::AbstractMatrix{T})
  ny = size(y,2)
  modelfit = [100*(1 - mse[i]/cov(y[1:N,i])) for i in 1:ny] # TODO fix to correct order m y[m:N]
end

function predict{T1,V1,V2,S,U,M,O}(data::IdDataObject{T1,V1,V2},
  model::PolyModel{S,U,M}, Θ, options::IdOptions{O}=IdOptions())
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
  icbf  = nbf > 0  ? reshape(Θᵢ[1:nbf*ny], nbf, ny)                  : zeros(T,0,0)
  icdc  = ndc > 0  ? reshape(Θᵢ[nbf*ny+(1:ndc*ny)], nbf, ny)         : zeros(T,0,0)
  iccda = ncda > 0 ? reshape(Θᵢ[(nbf+ndc)*ny+(1:ncda*ny)], ncda, ny) : zeros(T,0,0)
  return Θₚ, icbf, icdc, iccda
end

# calculate the value function V. Used for automatic differentiation
function cost{T<:Real,S,U,M,O}(data::IdDataObject{T}, model::PolyModel{S,U,M}, x,
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
  sumvalue(options.loss_function, y, y_est)/(2N)
