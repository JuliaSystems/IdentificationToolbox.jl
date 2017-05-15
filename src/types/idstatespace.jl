immutable IdStateSpace{T,S,M,I<:IdInfo} <: SystemsBase.LtiSystem
  A::M
  B::M
  C::M
  D::M
  K::M
  nx::Int
  nu::Int
  ny::Int
  Sigma::M
  Ts::Float64
  info::I

  @compat function (::Type{IdStateSpace}){M,I}(A::M, B::M,
    C::M, D::M, K::M, Sigma::M, Ts::Float64, info::I)
    na  = size(A,1)
    nb  = size(B,2)
    nc  = size(C,1)
    return new{Val{:mimo},Val{:disc},M,I}(A, B, C, D, K, na, nb, nc, Sigma, Ts, info)
  end

  @compat function (::Type{IdStateSpace}){M,I}(A::M, B::M,
    C::M, d::Real, K::M, Sigma::M, Ts::Float64, info::I)
    na  = size(A,1)
    nb  = size(B,2)
    nc  = size(C,1)
    return new{Val{:siso},Val{:disc},M,I}(A, B, C, fill(d,1,1), K, na, nb, nc, Sigma, Ts, info)
  end
end

# conversion to state-space
ss(s::IdStateSpace{Val{:siso},Val{:disc}}) = ss(s.A, s.B, s.C, s.D[1], s.Ts)
ss(s::IdStateSpace{Val{:mimo},Val{:disc}}) = ss(s.A, s.B, s.C, s.D, s.Ts)

samplingtime(s::IdStateSpace) = s.Ts
isdiscrete(s::IdStateSpace)   = true

summary(s::IdStateSpace) = string("ss(nx=", s.nx, ",nu=", s.nu, ",ny=", s.ny, ",Ts=",
  s.Ts, ")")

showcompact(io::IO, s::IdStateSpace) = print(io, summary(s))

function show(io::IO, s::IdStateSpace)
  println(io, "Discrete time state space model")
  println(io, "\tx[k+1] = Ax[k] + Bu[k]")
  println(io, "\ty[k]   = Cx[k] + Du[k]")
  print(io, "with nx=", s.nx, ", nu=", s.nu, ", ny=", s.ny, ", Ts=", s.Ts, ".")
end

function showall(io::IO, s::IdStateSpace)
  show(io, s)
  println(io, "System matrix (A):")
  println(io, s.A)
  println(io, "Input matrix (B):")
  println(io, s.B)
  println(io, "Output matrix (C):")
  println(io, s.C)
  println(io, "Feedforward matrix (D):")
  print(io, s.D)
end
