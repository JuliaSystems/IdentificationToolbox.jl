@compat abstract type AbstractModelOrder end

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
    new{Val{:mimo}}(na,nb,nf,nc,nd,nk)
  end

  @compat function (::Type{FullPolyOrder})(na::Int, nb::Int, nf::Int, nc::Int,
    nd::Int, nk::Int)
    _fullpolyordercheck(na,nb,nf,nc,nd,nk)
    new{Val{:siso}}(na,nb,nf,nc,nd,[nk])
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

function getindex(p::FullPolyOrder{Val{:siso}}, row::Int, col::Int)
  1 ≤ col ≤ length(p.nk) || error("s[,idx]: idx out of bounds")
  FullPolyOrder(p.na, p.nb, p.nf, p.nc, p.nd, p.nk[1])
end

function getindex(p::FullPolyOrder{Val{:mimo}}, row::Int, col::Int)
  1 ≤ col ≤ length(p.nk) || error("s[,idx]: idx out of bounds")
  FullPolyOrder(p.na, p.nb, p.nf, p.nc, p.nd, p.nk)
end

getindex{S}(p::FullPolyOrder{S}, ::Colon)           = p
getindex{S}(p::FullPolyOrder{S}, ::Colon, ::Colon)  = p
getindex{S}(p::FullPolyOrder{S}, ::Colon, idx::Int) = p[1,idx]
getindex{S}(p::FullPolyOrder{S}, idx::Int, ::Colon) = p

@compat abstract type IdModel{S} end

immutable SSModel{S} <: IdModel{S}
  order::Int
  ny::Int
  nu::Int

  @compat function (::Type{SSModel}){S}(order::Integer, ny::Integer, nu::Integer, ::Type{Val{S}})
    new{Val{S}}(order,ny,nu)
  end
end

@compat abstract type PolyType end

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

immutable PolyModel{S,M,P} <: IdModel{S}
  orders::M
  ny::Int
  nu::Int

  @compat function (::Type{PolyModel}){M<:AbstractModelOrder,
      P<:PolyType, S}(orders::M, ny::Int, nu::Int, ::Type{Val{S}}, ::Type{P})
    new{Val{S},M,P}(orders,ny,nu)
  end
end

size(p::PolyModel) = ny,nu
function size{S}(p::PolyModel{S}, i::Int)
  i > 0 || throw(ArgumentError("size: dimension needs to be postive"))
  if i == 1
    return ny
  elseif i == 2
    return nu
  end
  return 1
end
length(p::PolyModel) = p.ny*p.nu

# Slicing (`getindex`)
function getindex{S,M,P}(p::PolyModel{S,M,P}, row::Int, col::Int)
  1 ≤ row ≤ p.ny || error("s[idx,]: idx out of bounds")
  1 ≤ col ≤ p.nu || error("s[,idx]: idx out of bounds")
  PolyModel(p.orders[row,col], 1, 1, S, P)
end

getindex{S,M,P}(p::PolyModel{S,M,P}, ::Colon)           = p
getindex{S,M,P}(p::PolyModel{S,M,P}, ::Colon, ::Colon)  = p

function getindex{S,M,P}(p::PolyModel{S,M,P}, ::Colon, idx::Int)
  PolyModel(p.orders[:,idx], ny, 1, S, P)
end

function getindex{S,M,P}(p::PolyModel{S,M,P}, idx::Int, ::Colon)
  PolyModel(orders[idx,:], 1, nu, S, P)
end

# outer constructors

"""
    `PolyModel(na, nb, nf, nc, nd, nk=1)`

Define the PolyModel(`na`,`nb`,`nf`,`nc`,`nd`) model structure:
    A(z)y(t) = F(z)\B(z) z^-`nk`u(t) + D(z)\C(z)e(t)
"""
# TODO: reconsider
function PolyModel(; na::Int=0, nb::Int=0, nf::Int=0, nc::Int=0, nd::Int=0,
  nk::Union{Int,Vector{Int}}=1, ny::Int=1, nu::Int=1)
  isa(nk, Int) ? _PolyModel(Val{:siso}; na=na, nb=nb, nf=nf, nc=nc, nd=nd, nk=[nk], ny=ny, nu=nu) :
                 _PolyModel(Val{:mimo}; na=na, nb=nb, nf=nf, nc=nc, nd=nd, nk=nk, ny=ny, nu=nu)
end

function _PolyModel{S}(::Type{Val{S}}; na::Int=0,nb::Int=0,nf::Int=0,nc::Int=0,nd::Int=0,
  nk::Vector{Int}=[1],ny::Int=1,nu::Int=1)
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
  PolyModel(orders, ny, nu, Val{S}, CUSTOM)
end

"""
    `FIR(nb, nk=1)`

Define the FIR(`nb`) model structure:
    y(t) = B(z) z^-`nk`u(t) + e(t)
"""
function FIR(nb::Int, nk::Int, ny::Int=1, nu::Int=1)
  _FIR(nb, [nk], ny, nu)
end

function FIR(nb::Int, nk::Vector{Int}, ny::Int=1, nu::Int=1)
  _FIR(nb, nk, ny, nu)
end

function _FIR{S}(::Type{Val{S}}, nb::Int, nk::Vector{Int}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(0, nb, 0, 0, 0, nk)
  PolyModel(orders, ny, nu, Val{S}, FIR)
end

function FIR{M<:AbstractMatrix{Int}}(nb::M, nk::M, ny::Int=1, nu::Int=1)
  #_delaycheck(nk, nu)
  orders = MPolyOrder(zeros(Int,ny,ny), nb, zeros(Int,ny,nu), zeros(Int,ny,ny), zeros(Int,ny,ny), nk)
  PolyModel(orders, ny, nu, Val{:mimo}, FIR)
end

"""
    `AR(na, ny=1, nu=1)`

Define the AR(`na`) model structure :
    A(z)y(t) =  e(t)
"""
function AR(na::Int, ny::Int=1, nu::Int=0)
  orders = FullPolyOrder(na, 0, 0, 0, 0, zeros(Int,nu))
  ny > 1 ? PolyModel(orders, ny, nu, Val{:mimo}, AR) :
           PolyModel(orders, ny, nu, Val{:siso}, AR)
end

function AR{M<:AbstractMatrix{Int}}(na::M, ny::Int=1, nu::Int=1)
  #_delaycheck(nk, nu)
  orders = MPolyOrder(na, zeros(Int,ny,nu), zeros(Int,ny,nu), zeros(Int,ny,ny), zeros(Int,ny,ny), zeros(Int,nu))
  PolyModel(orders, ny, nu, Val{:mimo}, AR)
end

"""
    `ARX(na, nb, nk=1)`

Define the ARX(`na`,`nb`) model structure:
    A(z)y(t) = B(z) z^-`nk`u(t) + e(t)
"""
function ARX(na::Int, nb::Int, nk::Int, ny::Int=1, nu::Int=1)
  _ARX(Val{:siso}, na, nb, [nk], ny, nu)
end

function ARX(na::Int, nb::Int, nk::Vector{Int}, ny::Int=1, nu::Int=1)
  _ARX(Val{:mimo}, na, nb, nk, ny, nu)
end

function _ARX{S}(::Type{Val{S}}, na::Int, nb::Int, nk::Vector{Int}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(na, nb, 0, 0, 0, nk)
  PolyModel(orders, ny, nu, Val{S}, ARX)
end

function ARX{M<:AbstractMatrix{Int}}(na::M, nb::M, nk::M, ny::Int=1, nu::Int=1)
  #_delaycheck(nk, nu)
  orders = MPolyOrder(na, nb, zeros(Int,ny,nu), zeros(Int,ny,ny), zeros(Int,ny,ny), nk)
  PolyModel(orders, ny, nu, Val{:mimo}, ARX)
end

# """
#     `ARMAX(na, nb, nc, nk=1)`
#
# Define the ARMAX(`na`,`nb`,`nc`) model structure:
#     A(z)y(t) = B(z) z^-`nk`u(t) + C(z)e(t)
# """
# function ARMAX(na::Int, nb::Int, nc::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
#   _delaycheck(nk, nu)
#   orders = FullPolyOrder(na, nb, 0, nc, 0, nk)
#   S = (typeof(nk) == Int)
#   PolyModel(orders, ny, nu, ControlCore.Siso{S}, ARMAX)
# end
#
# function ARMAX{M<:AbstractMatrix{Int}}(na::M, nb::M, nc::M, nk::M, ny::Int=1, nu::Int=1)
#   #_delaycheck(nk, nu)
#   orders = MPolyOrder(na, nb, zeros(Int,ny,nu), nc, zeros(Int,ny,ny), nk)
#   PolyModel(orders, ny, nu, Val{:mimo}, ARMAX)
# end
#
# """
#     `ARMA(na, nc, ny=1, nu=1)`
#
# Define the ARMA(`na`,`nc`) model structure :
#     A(z)y(t) =  C(z) e(t)
# """
# function ARMA(na::Int, nc::Int, ny::Int=1, nu::Int=1)
#   orders = FullPolyOrder(na, 0, 0, nc, 0, zeros(Int,nu))
#   S = (typeof(nk) == Int)
#   PolyModel(orders, ny, nu, ControlCore.Siso{S}, ARMA)
# end
#
# """
#     `ARARX(na, nb, nd, nk, ny=1, nu=1)`
#
# Define the ARARX(`na`,`nb`,`nd`,`nk`) model structure :
#     A(z)y(t) = B(z)z^-`nk`u(t) + D(z)\1 e(t)
# """
# function ARARX(na::Int, nb::Int, nd::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
#   _delaycheck(nk, nu)
#   orders = FullPolyOrder(na, nb, 0, 0, nd, nk)
#   S = (typeof(nk) == Int)
#   PolyModel(orders, ny, nu, ControlCore.Siso{S}, ARARX)
# end

"""
    `oe(nb, nf, nk, ny=1, nu=1)`

Define the OE(`nb`,`nf`,`nk`) model structure :
    y(t) = F(z)\B(z)z^-`nk`u(t) + e(t)
"""
function OE(nb::Int, nf::Int, nk::Int, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(0, nb, nf, 0, 0, nk)
  PolyModel(orders, ny, nu, Val{:siso}, OE)
end

function OE(nb::Int, nf::Int, nk::Vector{Int}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(0, nb, nf, 0, 0, nk)
  PolyModel(orders, ny, nu, Val{:mimo}, OE)
end

function OE{M<:AbstractMatrix{Int}}(nb::M, nf::M, nk::M, ny::Int=1, nu::Int=1)
  #_delaycheck(nk, nu)
  orders = MPolyOrder(zeros(Int,ny,ny), nb, nf, zeros(Int,ny,ny), zeros(Int,ny,ny), nk)
  PolyModel(orders, ny, nu, Val{:mimo}, OE)
end

# """
#     `ARARMAX(na, nb, nc, nd, nk, ny=1, nu=1)`
#
# Define the ARARMAX(`na`,`nb`,`nc`,`nd`,`nk`) model structure :
#     A(z)y(t) = B(z)z^-`nk`u(t) + D(z)\C(z) e(t)
# """
# function ARARMAX(na::Int, nb::Int, nc::Int, nd::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
#   _delaycheck(nk, nu)
#   orders = FullPolyOrder(na, nb, 0, nc, nd, nk)
#   S = (typeof(nk) == Int)
#   PolyModel(orders, ny, nu, ControlCore.Siso{S}, ARARMAX)
# end

"""
    `BJ(nb, nc, nd, nf, nk=1)`

Define the Box-Jenkins BJ(`nb`,`nc`,`nd`,`nf`,`nk`) model structure:
    y(t) = F(z)\B(z)z^-`nk`u(t) + D(z)\C(z) e(t)
"""
function BJ(nb::Int, nf::Int, nc::Int, nd::Int, nk::Int, ny::Int=1, nu::Int=1)
  _BJ(Val{:mimo}, nb, nf, nc, nd, [nk], ny, nu)
end

function BJ(nb::Int, nf::Int, nc::Int, nd::Int, nk::Vector{Int}, ny::Int=1, nu::Int=1)
  _BJ(Val{:mimo}, nb, nf, nc, nd, nk, ny, nu)
end

function _BJ{S}(::Type{Val{S}}, nb::Int, nf::Int, nc::Int, nd::Int,
  nk::Vector{Int}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolyOrder(0, nb, nf, nc, nd, nk)
  PolyModel(orders, ny, nu, Val{S}, BJ)
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
  nc::Matrix{Int}
  nd::Matrix{Int}
  nk::Matrix{Int}

  @compat function MPolyOrder(na::Matrix{Int}, nb::Matrix{Int}, nf::Matrix{Int},
    nc::Matrix{Int}, nd::Matrix{Int}, nk::Matrix{Int})
    _mpolyordercheck(na,nb,nf,nc,nd,nk)
    new(na,nb,nf,nc,nd,nk)
  end
end

function _mpolyordercheck(na,nb,nf,nc,nd,nk)
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
  ny,nu = size(nb)
  size(na) == (ny,ny) || error("MPolyOrder: input dimensions do not match")
  size(nf) == (ny,nu) || error("MPolyOrder: input dimensions do not match")
  size(nc) == (ny,ny) || error("MPolyOrder: input dimensions do not match")
  size(nd) == (ny,ny) || error("MPolyOrder: input dimensions do not match")
  isdiag(nc)          || error("MPolyOrder: Noise filter needs to be diagonal")
  isdiag(nd)          || error("MPolyOrder: Noise filter needs to be diagonal")
end

size(p::MPolyOrder) = size(p.nb)
# Slicing (`getindex`)
function getindex(p::MPolyOrder, row::Int, col::Int)
  1 ≤ row ≤ size(p,1) || error("s[idx,]: idx out of bounds")
  1 ≤ col ≤ size(p,2) || error("s[,idx]: idx out of bounds")
  MPolyOrder(p.na[row:row,col:col], p.nb[row:row,col:col], p.nf[row:row,col:col], p.nc[row:row,col:col], p.nd[row:row,col:col])
end

getindex(p::MPolyOrder, ::Colon)           = p
getindex(p::MPolyOrder, ::Colon, ::Colon)  = p

function getindex(p::MPolyOrder, ::Colon, idx::Int)
  1 ≤ idx ≤ size(p,2) || error("s[,idx]: idx out of bounds")
  PolyModel(p.na[:,idx:idx], p.nb[:,idx:idx], p.nf[:,idx:idx], p.nc[:,idx:idx], p.nd[:,idx:idx])
end

function getindex(p::MPolyOrder, idx::Int, ::Colon)
  PolyModel(p.na[idx:idx,:], p.nb[idx:idx,:], p.nf[idx:idx,:], p.nc[idx:idx,:], p.nd[idx:idx,:])
end
