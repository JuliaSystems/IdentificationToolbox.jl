immutable IdStateSpace{T,M,S<:IdInfo} <: SystemsBase.LtiSystem
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
  info::S

  @compat function (::Type{IdStateSpace}){M,S}(A::M, B::M,
    C::M, D::M, K::M, Sigma::M, Ts::Float64, info::S)
    na  = size(A,1)
    nb  = size(B,1)
    nc  = size(C,1)
    T   = eltype(A)
    return new{T,M,S}(A, B, C, D, K, na, nb, nc, Sigma, Ts, info)
  end
end

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
