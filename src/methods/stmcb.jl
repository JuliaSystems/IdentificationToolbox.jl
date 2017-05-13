function stmcb{T,A1,A2,S,U}(
    data::IdDataObject{T,A1,A2}, model::PolyModel{S,U,OE},
    options::IdOptions=IdOptions(estimate_initial=false),
    C::Matrix{T}=zeros(T,0,0), D::Matrix{T}=zeros(T,0,0))

  x         = _stmcb(data, model, options, C, D)
  mse       = _mse(data, model, x, options)
  modelfit  = _modelfit(mse, data.y)
  idinfo    = OneStepIdInfo(mse, modelfit, model)
  a,b,f,c,d = _getpolys(model, x)

  IdMFD(a, b, f, c, d, data.Ts, idinfo)
end

function _stmcb{T,A1,A2,S,U}(
    data::IdDataObject{T,A1,A2}, model::PolyModel{S,U,OE},
    options::IdOptions=IdOptions(estimate_initial=false),
    c::Matrix{T}=ones(T,0,0), d::Matrix{T}=ones(T,0,0))
  na,nb,nf,nc,nd,nk = orders(model)
  y,u         = data.y,data.u
  ny,nu       = data.ny,data.nu
  feedthrough = false
  iterations  = options.OptimizationOptions.iterations
  @assert !feedthrough || nk > zero(Int) string("nk must be greater than zero if feedthrough term is known")

  if ny != nu
    warn("The steglitz Mcbride method is currently only implemented for square systems")
    throw(DomainError())
  end

  # no known noise model
  if length(c) < 1
    c = eye(T,ny,ny)
    d = eye(T,ny,ny)
  end
  nc = convert(Int, round(size(c,1)/ny)-1)
  nd = convert(Int, round(size(d,1)/ny)-1)

  arxmodel = ARX(nf,nb,nk,ny,nu)
  bjmodel = feedthrough ? BJ(nb,nf,nc,nd,zeros(Int,nu),ny,nu) :
                          BJ(nb,nf,nc,nd,nk,ny,nu)

  # first iteration the data is not pre-filtered
  yf     = deepcopy(data.y)
  uf     = deepcopy(data.u)
  dataf  = iddata(yf, uf, data.Ts)

  bestb   = zeros(T,nb*nu,ny)
  bestf   = zeros(T,nf*ny,ny)
  bestpe = typemax(Float64)
  for i = 1:iterations
    Θ  = _arx(dataf, arxmodel, options)[1]
    Θᵣ = reshape(Θ, nb*nu+ny*nf, ny)
    f  = Θᵣ[1:nf*ny,1:ny]
    b  = Θᵣ[ny*nf+(1:nb*nu),1:ny]

    x  = vcat(b, f, c, d)
    pe = cost(data, bjmodel, x[:], options)

    if pe < bestpe
      bestb = b
      bestf = f
      bestpe = pe
    end

    # filter data
    xf    = _blocktranspose(f, ny, ny, nf)
    F     = PolyMatrix(vcat(eye(T,ny),xf), (ny,ny))
    _filt_ar!(yf, F, y)
    _filt_ar!(uf, F, u)
  end
  return vec(vcat(bestb, bestf))
end
