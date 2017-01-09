using ControlCore
using ControlToolbox
using IdentificationToolbox
using Polynomials
using Optim
using NetworkEmpiricalBayes

Ts = 1.
b1 = [0, .4, -.5]
a1 = [1., .3, .1]
b2 = [0, .4, .5]
a2 = [1., .5, .2]

G31 = tf([.0, .5], [1., 0.6], Ts, :qinv)
G32 = tf([.0, .9], [1., 0.5], Ts, :qinv)

s11 = tf(b1,a1,Ts)
s12 = tf(b1,a1,Ts)
s21 = tf(b1,a1,Ts)
s22 = tf(b1,a1,Ts)
G31.num

N = 20
r1  = randn(N)
r2  = randn(N)
u10 = filt(b1,a1,r1) + filt(b2,a1,r2)
u20 = filt(b2,a2,r1) + filt(b1,a2,r2)
u   = hcat(u10,u20)  + 1*randn(N,2)
y   = filt([.0, .5], [1., 0.6], u10) + filt([.0, .9], [1., 0.6], u20) + 0.1*randn(N)
r   = hcat(r1,r2)


NEB(y,u,r,15,1)


options = IdOptions(g_tol = 1e-8, extended_trace=false, iterations = 10, loss_function = L2DistLoss(), autodiff=true, estimate_initial=false)
m1    = FIR(15,[1,1],1,2)
# S1
data1 = iddata(u[:,1],r)
S1h = pem(data1,m1,zeros(30),options=options)

# S2
data2 = iddata(u[:,2],r)
S2h = pem(data2,m1,zeros(30),options=options)

u1h = filt(S1h.B,S1h.F,r)
u2h = filt(S2h.B,S2h.F,r)

uh = hcat(u1h,u2h)
data3 = iddata(y,uh,Ts)

options = IdOptions(g_tol = 1e-8, extended_trace=false, iterations = 10, loss_function = L2DistLoss(), autodiff=true, estimate_initial=false)
m2    = OE(1,1,[1,1],1,2)
x0 = vcat([0.1,0.5,0.1,0.6])

Gh = pem(data3,m2,x0,options=options)
Gh.B
Gh.F

S1h.info.opt.minimizer
coeffs(S1h.B[1])
data = iddata(hcat(y,u),r)
model = [15,1,1]
x = vcat(coeffs(S1h.B[1])[2:end], coeffs(S1h.B[2])[2:end], coeffs(S2h.B[1])[2:end], coeffs(S2h.B[2])[2:end])
x0 = vcat(x, coeffs(Gh.B[1])[2:end], coeffs(Gh.F[1])[2:end], coeffs(Gh.B[2])[2:end], coeffs(Gh.F[1])[2:end])


options = IdOptions(g_tol = 1e-16, extended_trace=false, iterations = 100, loss_function = L2DistLoss(), autodiff=true, estimate_initial=false)
cost(data,model,x0,options)

opt = optimize(x->cost(data, model, x, options), x0, Newton(), options.OptimizationOptions)

opt.minimizer
V = cost(data, model, opt.minimizer, options)


NEB(y,u,r,15,1)

import IdentificationToolbox.cost

function cost{T1<:Real,T2,O}(data::IdentificationToolbox.IdDataObject{T1},
    model::Vector{Int}, x::AbstractArray{T2},
    options::IdentificationToolbox.IdOptions{O}=IdOptions())
  T = promote_type(T1,T2)
  nu,nr = data.ny-1,data.nu
  y     = data.y[:,1]
  u     = data.y[:,2:nu+1]
  r     = data.u
  N     = size(y,1)

  nb = model[1]
  noeb = model[2]
  noef = model[3]
  ns = nu*nr

  s = x[1:nb*ns]
  uhat = zeros(T,N,nu)
  for i = 1:nu
    for k = 1:nr
      m = (i-1)*nr*nb + (k-1)*nb
      uhat[:,i] += filt(vcat(zero(T),s[m+(1:nb)]),1,r[:,k])
    end
  end

  yhat = zeros(T,N)
  g = x[nu*nr*nb+(1:(noeb+noef)*nu)]
  for i = 1:nu
    m = (i-1)*(noeb+noef)
    yhat += filt(g[m+(1:noeb)], g[m+noeb+(1:noef)],uhat[:,i])
  end

  ycost = IdentificationToolbox.cost(y, yhat, N, options)
  ucost = IdentificationToolbox.cost(u, uhat, N, options)
  #println(ucost)
  #println(ycost)
  return ycost + ucost #, yhat
end
