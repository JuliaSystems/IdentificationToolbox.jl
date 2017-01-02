println("Starting OE method test...")

# Define the true system
b = [0.1, 0.3, 0.2]
f = [0.5, 0.11, 0.05]

# model orders
nb, nf = 30, 30
nk = 1
n = [nb, nf, nk]

# intitial parameters
x = [b; zeros(nb-3);f; zeros(nf-3)]

# generate input data+noise and simulate output
B = [0;b]
F = [1;f]
N = 1000
u = randn(N)
lambda = 10
e = sqrt(lambda)*randn(N)
y = filt(B,F,u,[2.;2;2]) + e

# create iddataObject for the input/output data
data = iddata(y,u)

# test internal method
k = nb+nf
storage = zeros(Float64,k,k+1)
last_V = [-0.1]
V = IdentificationToolbox.gradhessian!(data, n, x, OE(), 0.9*x, last_V, storage)
@test abs(V-lambda) < 0.3*lambda

# test user methods
S = Array(IdDSisoRational,0)

# bj constructor
push!(S, oe(data, n, x))

# pem constructor
push!(S, pem(data, n, x, OE()))

for system in S
  @test abs(system.info.mse-lambda) < 0.3*lambda
end


b = [0.3]
f = [0.5]

B = [0;b]
F = [1;f]

nb = nf = 1

N = 2000
u1 = randn(N)
u2 = randn(N)
lambda = 1
e1 = sqrt(lambda)*randn(N)
e2 = sqrt(lambda)*randn(N)
y1 = filt(B,F,u1) + filt(B,F,u2) + e1
y2 = 0*filt(B,F,u1) + filt(B,F,u2) + e2

u = hcat(u1,u2)
y = hcat(y1)
data2 = iddata(y, u)

ny = 1
nu = 2

model = OE(nb,nf,1,ny,nu)
model2 = OE(nb,nf,[1],ny,nu)
orders(model)

m = nf*ny^2+nb*nu*ny
randn(m)
m0 = max(nb,nf)*ny

cost(data2, model, randn(m+m0))
_mse(data2, model, randn(m+m0))

psi = psit(data2, model, randn(m+m0))
orders(model)

x0 = vcat(b[1]*ones(nu,ny), f[1]*eye(ny))[:]
#x0 = vcat(b[1]*ones(nu,ny), b[2]*ones(nu,ny), b[3]*ones(nu,ny), f[1]*eye(ny), f[2]*eye(ny), f[3]*eye(ny))[:]

x0[2] = 0.5
x0 = vcat(x0,zeros(m0))
ap,bp,fp,cp,dp = _getpolys(model,x0)
ap,bp2,fp2,cp,dp = _getpolys(model2,x0)


filt(bp2,fp2,u)
filt(bp,fp,u)
filt(1,fp,u)
filt(Poly([1.]),fp,u)

cost(data2, model, x0, options)

options = IdOptions(f_tol=1e-64, extended_trace=false, iterations = 100, autodiff=false, show_trace=true, estimate_initial=true)
@time sys = pem(data2, model, x0 + 0.1*randn(length(x0)), options=options) # , IdOptions(f_tol = 1e-32)
sys.B
sys.F
sys.info.opt
fieldnames(sys.info.opt.trace[1].metadata)
sys.info.opt.trace[1]

sys.info.opt.trace[1].metadata.vals

options.estimate_initial

df = TwiceDifferentiableFunction(x->cost(data2, model, x0, options))
stor = zeros(2m,2m)
a = df.h!(x0,stor)
df

k = length(x0) # number of parameters
last_x  = zeros(Float64,k)
last_V  = -ones(Float64,1)
storage = zeros(k, k+1)
g = x0
@time gradhessian!(data2, model, x0, last_x, last_V, storage, options)
g = storage[:,end]
H = storage[:,1:end-1]
H\g
x = x0-0.0001*(H\g)

gradhessian!(data2, model, x, last_x, last_V, storage, options)
g = storage[:,end]
H = storage[:,1:end-1]+eye(k)*0.0001
x = x-0.1*(H\g)

psi = psit(data2, model, x, options)
psi.'*psi/N

@time y_est = predict(data2, model, x0)
predict(data2, model, x0)



model = ARMAX(1,1,1,[1,1],ny,nu)
orders(model)
x0 = vcat(f[1]*ones(nu,ny), b[1]*eye(ny), 0.1*eye(ny))[:]
m = length(x0)

@time sys = pem(data2, model, x0+0.05*randn(m))
sys.A
sys.B
sys.C
sys.info.opt

model = BJ(1,1,1,1,[1,1],ny,nu)
orders(model)
x0 = vcat(b[1]*ones(nu,ny), f[1]*eye(ny), 0.1*eye(ny), 0.1*eye(ny))[:]
m = length(x0)

@time sys = pem(data2, model, x0+0.05*randn(m))
sys.A
sys.B
sys.F
sys.C
sys.D
sys.info.opt

model = FIR(8,1,ny,nu)
orders(model)
x0 = vcat(b[1]*ones(nu,ny), 0.0*ones(7*nu,ny))[:]
m = length(x0)

x0 = vcat(x0, zeros(m))

options = IdOptions(f_tol=1e-64, extended_trace=false, iterations = 100, autodiff=true, show_trace=true, estimate_initial=false)
@time sys = pem(data2, model, x0, options=options)
sys.B
sys.info.opt



model = ARARX(1,1,1,1,ny,nu)
x0 = vcat(f[1]*eye(ny), b[1]*ones(nu,ny), 0*ones(1*ny,ny), 0*ones(1*ny,ny))[:]
m = length(x0)

na,nb,nf,nc,nd = orders(model)
nbf  = max(nb, nf)
ndc  = max(nd, nc)
ncda = max(nc, nd+na)
m0 = nbf+ndc+ncda
x0 = vcat(x0,zeros(m0))

options = IdOptions(f_tol=1e-64, extended_trace=false, iterations = 100, autodiff=true, show_trace=true, estimate_initial=true)
@time sys = pem(data2, model, x0+0.1*randn(length(x0)))
sys.B
sys.A
sys.info.opt

_split_params(model, x0, options)
_getpolys(model, x0[1:4])

na = nb = m = 10
model = ARX(m,m,[1,1],ny,nu)
x0 = zeros(ny^2*na+nb*ny*nu)
m = length(x0)

na,nb,nf,nc,nd = orders(model)
nbf  = max(nb, nf)
ndc  = max(nd, nc)
ncda = max(nc, nd+na)
m0 = nbf+ndc+ncda
x0 = vcat(x0,zeros(m0))

options = IdOptions(f_tol=1e-64, extended_trace=false, iterations = 100, autodiff=true, show_trace=true, estimate_initial=false)
@time sys = pem(data2, model, x0+0.1*randn(length(x0)), options=options)
sys.B
sys.A
sys.info.opt
