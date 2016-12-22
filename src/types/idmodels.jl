abstract AbstractModelOrder

immutable FullPolynomialOrder{S} <: AbstractModelOrder
  na::Int
  nb::Int
  nf::Int
  nc::Int
  nd::Int
  nk::Vector{Int}

  @compat function (::Type{FullPolynomialOrder})(na::Int, nb::Int, nf::Int, nc::Int,
    nd::Int, nk::Vector{Int})
    new{ControlCore.Siso{false}}(na,nb,nf,nc,nd,nk)
  end

  @compat function (::Type{FullPolynomialOrder})(na::Int, nb::Int, nf::Int, nc::Int,
    nd::Int, nk::Int)
    new{ControlCore.Siso{true}}(na,nb,nf,nc,nd,[nk])
  end
end

abstract IdModel

abstract PolynomialType

immutable FIR     <: PolynomialType end
immutable ARX     <: PolynomialType end
immutable ARMAX   <: PolynomialType end
immutable ARMA    <: PolynomialType end
immutable ARARX   <: PolynomialType end
immutable ARARMAX <: PolynomialType end
immutable OE      <: PolynomialType end
immutable BJ      <: PolynomialType end
immutable CUSTOM  <: PolynomialType end

immutable PolynomialModel{M,P} <: IdModel
  orders::M
  ny::Int
  nu::Int

  @compat function (::Type{PolynomialModel}){M<:AbstractModelOrder,P<:PolynomialType}(
    orders::M, ny::Int, nu::Int, ::Type{P})
    new{M,P}(orders,ny,nu)
  end
end

# outer constructors

function PolynomialModel(;na::Int=0,nb::Int=0,nf::Int=0,nc::Int=0,nd::Int=0,
  nk::Union{Int,Vector{Int}}=1,ny::Int=1,nu::Int=1)
  _delaycheck(nk, nu)
  na == nf == nc == nd == 0 && FIR(nb,nk)
  nf == nc == nd == 0       && ARX(na,nb,nk)
  na == nf == nd == 0       && ARMA(na,nc,nk)
  na == nc == nd == 0       && OE(nb,nf,nk)
  nf == nd == 0             && ARMAX(na,nb,nc,nk)
  nf == nc == 0             && ARARX(na,nb,nd,nk)
  nf == 0                   && ARARMAX(na,nb,nc,nd,nk)
  na == 0                   && BJ(nb,nf,nc,nd,nk)
  orders = FullPolynomialOrder(na, nb, nf, nc, nd, vcat(nk))
  PolynomialModel(orders, ny, nu, CUSTOM)
end

function FIR(nb::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolynomialOrder(0, nb, 0, 0, 0, nk)
  PolynomialModel(orders, ny, nu, FIR)
end

function ARX(na::Int, nb::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolynomialOrder(na, nb, 0, 0, 0, nk)
  PolynomialModel(orders, ny, nu, ARX)
end

function ARMAX(na::Int, nb::Int, nc::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolynomialOrder(na, nb, 0, nc, 0, nk)
  PolynomialModel(orders, ny, nu, ARMAX)
end

function ARMA(na::Int, nc::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolynomialOrder(na, nb, 0, nc, 0, nk)
  PolynomialModel(orders, ny, nu, ARMA)
end

function ARARX(na::Int, nb::Int, nd::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolynomialOrder(na, nb, 0, 0, nd, nk)
  PolynomialModel(orders, ny, nu, ARARX)
end

function OE(nb::Int, nf::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolynomialOrder(0, nb, nf, 0, 0, nk)
  PolynomialModel(orders, ny, nu, OE)
end

function ARARMAX(na::Int, nb::Int, nc::Int, nd::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolynomialOrder(na, nb, 0, nc, nd, nk)
  PolynomialModel(orders, ny, nu, ARARMAX)
end

function BJ(nb::Int, nf::Int, nc::Int, nd::Int, nk::Union{Int,Vector{Int}}, ny::Int=1, nu::Int=1)
  _delaycheck(nk, nu)
  orders = FullPolynomialOrder(0, nb, nf, nc, nd, nk)
  PolynomialModel(orders, ny, nu, BJ)
end

# FIR(nb::Int, nk::Int=1, ny::Int=1, nu::Int=1)                     = FIR(nb,[nk],ny,nu)
# ARX(na::Int, nb::Int, nk::Int=1, ny::Int=1, nu::Int=1)            = OE(na,nb,[nk],ny,nu)
# ARMAX(na::Int, nb::Int, nc::Int, nk::Int=1, ny::Int=1, nu::Int=1) = ARMAX(na,nb,nc,[nk],ny,nu)
# ARMA(na::Int, nc::Int, nk::Int=1, ny::Int=1, nu::Int=1)           = ARMA(na,nc,[nk],ny,nu)
# ARARX(na::Int, nb::Int, nd::Int, nk::Int=1, ny::Int=1, nu::Int=1) = ARARX(na,nb,nd,[nk],ny,nu)
# OE(nb::Int, nf::Int, nk::Int=1, ny::Int=1, nu::Int=1)             = OE(nb,nf,[nk],ny,nu)

# ARARMAX(na::Int, nb::Int, nc::Int, nd::Int, nk::Int=1, ny::Int=1, nu::Int=1) =
#   ARARMAX(na,nb,nc,nd,[nk],ny,nu)
# BJ(nb::Int, nf::Int, nc::Int, nd::Int, nk::Int=1, ny::Int=1, nu::Int=1) =
#   BJ(nb,nf,nc,nd,[nk],ny,nu)

function _delaycheck(nk, nu::Int)
  if length(nk) != nu
    warn("number of input delay specifications must match number of inputs")
    throw(DomainError())
  end
end

function orders(model::PolynomialModel)
  return orders(model.orders)
end

function orders(order::FullPolynomialOrder)
  return order.na, order.nb, order.nf, order.nc, order.nd, order.nk
end
