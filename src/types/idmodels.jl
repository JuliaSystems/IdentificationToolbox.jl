abstract AbstractModelOrder

immutable FullPolyOrder{S} <: AbstractModelOrder
  na::Int
  nb::Int
  nf::Int
  nc::Int
  nd::Int
  nk::Vector{Int}

  @compat function (::Type{FullPolyOrder})(na::Int, nb::Int, nf::Int, nc::Int,
    nd::Int, nk::Vector{Int})
    _fullpolyordercheck(na,nb,nf,nc,nd,nk)
    new{ControlCore.Siso{false}}(na,nb,nf,nc,nd,nk)
  end

  @compat function (::Type{FullPolyOrder})(na::Int, nb::Int, nf::Int, nc::Int,
    nd::Int, nk::Int)
    _fullpolyordercheck(na,nb,nf,nc,nd,nk)
    new{ControlCore.Siso{true}}(na,nb,nf,nc,nd,[nk])
  end
end

function _fullpolyordercheck(na,nb,nc,nd,nf,nk)
  for t in ((na,"na"),(nb,"nb"),(nf,"nf"),(nc,"nc"),(nd,"nd"))
    n,s = t
    if n < 0
      warn("FullPolyOrder: $s must be positive")
      throw(DomainError())
    end
  end
  if any(nk .< 0)
    warn("FullPolyOrder: nk must be positive")
    throw(DomainError())
  end
  if na+nb+nf+nc+nd < 1
    warn("FullPolyOrder: at least one model order must be greater than zero")
    throw(DomainError())
  end
end

abstract IdModel

abstract PolyType

immutable FIR     <: PolyType end
immutable AR      <: PolyType end
immutable ARX     <: PolyType end
immutable ARMAX   <: PolyType end
immutable ARMA    <: PolyType end
immutable ARARX   <: PolyType end
immutable ARARMAX <: PolyType end
immutable OE      <: PolyType end
immutable BJ      <: PolyType end
immutable CUSTOM  <: PolyType end

immutable PolyModel{S,M,P} <: IdModel
  orders::M
  ny::Int
  nu::Int

  @compat function (::Type{PolyModel}){M<:AbstractModelOrder,
      P<:PolyType, S}(orders::M, ny::Int, nu::Int, ::Type{S}, ::Type{P})
    new{S,M,P}(orders,ny,nu)
  end
end

# outer constructors

"""
    `PolyModel(na, nb, nf, nc, nd, nk=1)`

Define the PolyModel(`na`,`nb`,`nf`,`nc`,`nd`) model structure:
    A(z)y(t) = F(z)\B(z) z^-`nk`u(t) + D(z)\C(z)e(t)
"""
function PolyModel(;na::Int=0,nb::Int=0,nf::Int=0,nc::Int=0,nd::Int=0,
  nk::Union{Int,Vector{Int}}=1,ny::Int=1,nu::Int=1)
  _delaycheck(nk, nu)
  na == nf == nc == nd == 0 && FIR(nb,nk)
  nb == nf == nc == nd == 0 && AR(na)
  nf == nc == nd == 0       && ARX(na,nb,nk)
  na == nf == nd == 0       && ARMA(na,nc,nk)
  na == nc == nd == 0       && OE(nb,nf,nk)
  nf == nd == 0             && ARMAX(na,nb,nc,nk)
  nf == nc == 0             && ARARX(na,nb,nd,nk)
  nf == 0                   && ARARMAX(na,nb,nc,nd,nk)
  na == 0                   && BJ(nb,nf,nc,nd,nk)
  orders = FullPolyOrder(na, nb, nf, nc, nd, vcat(nk))
  S = (typeof(nk) == Int)
  PolyModel(orders, ny, nu, ControlCore.Siso{S}, CUSTOM)
end

"""
    `FIR(nb, nk=1)`

Define the FIR(`nb`) model structure:
    y(t) = B(z) z^-`nk`u(t) + e(t)
"""
function FIR(nb::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(0, nb, 0, 0, 0, nk)
  S = (typeof(nk) == Int)
  PolyModel(orders, ny, nu, ControlCore.Siso{S}, FIR)
end

"""
    `AR(na, ny=1, nu=1)`

Define the AR(`na`) model structure :
    A(z)y(t) =  e(t)
"""
function AR(na::Int, ny::Int=1, nu::Int=1)
  orders = FullPolyOrder(na, 0, 0, 0, 0, zeros(Int,nu))
  S = (typeof(nk) == Int)
  PolyModel(orders, ny, nu, ControlCore.Siso{S}, AR)
end

"""
    `ARX(na, nb, nk=1)`

Define the ARX(`na`,`nb`) model structure:
    A(z)y(t) = B(z) z^-`nk`u(t) + e(t)
"""
function ARX(na::Int, nb::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(na, nb, 0, 0, 0, nk)
  S = (typeof(nk) == Int)
  PolyModel(orders, ny, nu, ControlCore.Siso{S}, ARX)
end

"""
    `ARMAX(na, nb, nc, nk=1)`

Define the ARMAX(`na`,`nb`,`nc`) model structure:
    A(z)y(t) = B(z) z^-`nk`u(t) + C(z)e(t)
"""
function ARMAX(na::Int, nb::Int, nc::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(na, nb, 0, nc, 0, nk)
  S = (typeof(nk) == Int)
  PolyModel(orders, ny, nu, ControlCore.Siso{S}, ARMAX)
end

"""
    `ARMA(na, nc, ny=1, nu=1)`

Define the ARMA(`na`,`nc`) model structure :
    A(z)y(t) =  C(z) e(t)
"""
function ARMA(na::Int, nc::Int, ny::Int=1, nu::Int=1)
  orders = FullPolyOrder(na, 0, 0, nc, 0, zeros(Int,nu))
  S = (typeof(nk) == Int)
  PolyModel(orders, ny, nu, ControlCore.Siso{S}, ARMA)
end

"""
    `ARARX(na, nb, nd, nk, ny=1, nu=1)`

Define the ARARX(`na`,`nb`,`nd`,`nk`) model structure :
    A(z)y(t) = B(z)z^-`nk`u(t) + D(z)\1 e(t)
"""
function ARARX(na::Int, nb::Int, nd::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(na, nb, 0, 0, nd, nk)
  S = (typeof(nk) == Int)
  PolyModel(orders, ny, nu, ControlCore.Siso{S}, ARARX)
end

"""
    `oe(nb, nf, nk, ny=1, nu=1)`

Define the OE(`nb`,`nf`,`nk`) model structure :
    y(t) = F(z)\B(z)z^-`nk`u(t) + e(t)
"""
function OE(nb::Int, nf::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(0, nb, nf, 0, 0, nk)
  S = (typeof(nk) == Int)
  PolyModel(orders, ny, nu, ControlCore.Siso{S}, OE)
end

"""
    `ARARMAX(na, nb, nc, nd, nk, ny=1, nu=1)`

Define the ARARMAX(`na`,`nb`,`nc`,`nd`,`nk`) model structure :
    A(z)y(t) = B(z)z^-`nk`u(t) + D(z)\C(z) e(t)
"""
function ARARMAX(na::Int, nb::Int, nc::Int, nd::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(na, nb, 0, nc, nd, nk)
  S = (typeof(nk) == Int)
  PolyModel(orders, ny, nu, ControlCore.Siso{S}, ARARMAX)
end

"""
    `BJ(nb, nc, nd, nf, nk=1)`

Define the Box-Jenkins BJ(`nb`,`nc`,`nd`,`nf`,`nk`) model structure:
    y(t) = F(z)\B(z)z^-`nk`u(t) + D(z)\C(z) e(t)
"""
function BJ(nb::Int, nf::Int, nc::Int, nd::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(0, nb, nf, nc, nd, nk)
  S = (typeof(nk) == Int)
  PolyModel(orders, ny, nu, ControlCore.Siso{S}, BJ)
end

function _delaycheck(nk, nu::Int)
  if length(nk) != nu
    warn("number of input delay specifications must match number of inputs")
    throw(DomainError())
  end
end

function orders{P<:PolyModel}(model::P)
  return orders(model.orders)
end

function orders{O<:AbstractModelOrder}(order::O)
  return order.na, order.nb, order.nf, order.nc, order.nd, order.nk
end

#= Matlab MIMO model
A(q)y = B(q)/F(q)u + C(q)/D(q)e
A(q) is polynomial matrix of size ny×ny with A(q) = I + A₁q^(-1) + A₂q^(-2) ...
B(q) and F(q) are matrices of size ny×nu of SISO Polynomials representing the entries of the
numerator and denominator of the matrix of SISO transfer functions B(q)/F(q)
C(q) and D(q) are diagonal polynomial matrices
=#

immutable MPolyOrder <: AbstractModelOrder
  na::Matrix{Int}
  nb::Matrix{Int}
  nf::Matrix{Int}
  nc::Vector{Int}
  nd::Vector{Int}
  nk::Matrix{Int}

  @compat function MPolyOrder(na::Matrix{Int}, nb::Matrix{Int}, nf::Matrix{Int},
    nc::Vector{Int}, nd::Vector{Int}, nk::Matrix{Int})
    _mpolyordercheck(na,nb,nf,nc,nd,nk)
    new(na,nb,nf,nc,nd,nk)
  end
end

function _mpolyordercheck(na,nb,nc,nd,nf,nk)
  for t in ((na,"na"),(nb,"nb"),(nf,"nf"),(nc,"nc"),(nd,"nd"))
    n,s = t
    if any(n .< 0)
      warn("MPolyOrder: $s must be positive")
      throw(DomainError())
    end
  end
  if any(nk .< 0)
    warn("MPolyOrder: nk must be positive")
    throw(DomainError())
  end
  if sum(na.+nb.+nf.+nc.+nd) < 1
    warn("FullPolyOrder: at least one model order must be greater than zero")
    throw(DomainError())
  end
end
