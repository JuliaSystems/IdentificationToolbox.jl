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

B = [0; b]
F = [1; f]

nb = nf = 1

N  = 1000
u1 = randn(N)
u2 = randn(N)
lambda = 0.1
e1 = sqrt(lambda)*randn(N)
e2 = sqrt(lambda)*randn(N)
y1 = filt(B,F,u1) + 0.*filt(B,F,u2) + e1
y2 = 0.1*filt(B,F,u1) + filt(B,F,u2) + e2

u = hcat(u1)
y = hcat(y1)
data2 = iddata(y, u)

ny = size(u,2)
nu = size(y,2)

model = OE(nb,nf,[1],ny,nu)
model2 = OE(nb,nf,[1],ny,nu)
orders(model)

m = nf*ny^2+nb*nu*ny
randn(m)
m0 = max(nb,nf)*ny

cost(data2, model, randn(m+m0))
_mse(data2, model, randn(m+m0))

psi = psit(data2, model, randn(m+m0))
orders(model)

x0 = vcat(b[1]*ones(nu,ny), zeros((nb-1)*nu,ny), f[1]*eye(ny), zeros((nf-1)*ny,ny))[:]
#x0 = vcat(b[1]*ones(nu,ny), b[2]*ones(nu,ny), b[3]*ones(nu,ny), f[1]*eye(ny), f[2]*eye(ny), f[3]*eye(ny))[:]
x0 = vcat(x0,zeros(m0))

options = IdOptions(extended_trace=false, iterations = 10, autodiff=false, show_trace=true, estimate_initial=false)
cost(data2, model, x0, options)

# @time sys1 = pem(data2, model, x0 + 0.1*randn(length(x0)), options) # , IdOptions(f_tol = 1e-32)
# sys1.B
# sys1.F
# sys1.info.opt
# sys1.info.mse
# fieldnames(sys1.info.opt.trace[1].metadata)
# sys1.info.opt.trace[1]

options2 = IdOptions(f_tol=1e-64, extended_trace=false, iterations = 1, autodiff=true, show_trace=true, estimate_initial=false)
#_stmcb(data2,model,options2)
@time sys2 = stmcb(data2,model,options2)
sys2.B
sys2.F
sys2.info.mse

@time x,pe = _morsm(data2,model,options2)

@time begin
  m = 50
  Θₗ  = _arx(data2, ARX(m,m,1), options2)[1]
  xr  = reshape(Θₗ, ny*m+nu*m, ny) # [1:(ny*m+nu*m)*ny]
  xaₗ = view(xr, 1:ny*m, :)
  xbₗ = view(xr, ny*m+(1:nu*m), :)
  Aₗ  = PolyMatrix(vcat(eye(ny),      _blocktranspose(xaₗ, ny, ny, m)), (ny,ny))
  Bₗ  = PolyMatrix(vcat(zeros(ny,nu), _blocktranspose(xbₗ, ny, nu, m)), (ny,nu))
end

out = similar(y)
@time _filt_fir!(out,Bₗ,y)
y2 = randn(1,N)
out2 = similar(y2)
@time _filt_fir2!(out2,Bₗ,y2)

uf = similar(u)
@time _filt_fir!(uf, Aₗ, u)

bjmodel = BJ(nb,nf,0,12,1,ny,nu)


x2 = vcat(x[1],x[2], zeros(12))
cost(data2,bjmodel,x[:],options2)
cost(data2,model,x2,options2)
_mse(data2,model,x[1:2],options2)

uz    = zeros(100,nu)
uz[1] = 1.0
g2 = filt(sys2.B,sys2.F,uz)
g1 = filt(sys1.B,sys1.F,uz)
gt = filt(B,F,uz)

norm(g2-gt)
norm(g1-gt)

b2 = zeros(ny,nu*nb)
for i = 1:nb
  b2[:,(i-1)*nu+(1:nu)] = sys2.B.coeffs[i][:,:].'
end
f2 = zeros(ny,ny*nf)
for i = 1:nf
  f2[:,(i-1)*ny+(1:ny)] = sys2.F.coeffs[i][:,:].'
end
x02 = vcat(b2.',f2.')[:]

sumabs2(g1-gt)
sumabs2(g2-gt)
norm(g2-gt)
norm(g1-gt)

@time sys1 = pem(data2, model, x02, options) # , IdOptions(f_tol = 1e-32)
@time sys1 = pem(data2, model, sys1.info.opt.minimizer, options)
sys1.B
sys1.F
sys1.info.opt
sys1.info.mse

sys.info.opt.trace[1].metadata.vals

options.OptimizationOptions.iterations

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



model = OE(1,1,[1],ny,nu)
orders(model)
x0 = vcat(f[1]*ones(nu,ny), b[1]*eye(ny), 0.1*eye(ny))[:]
m = length(x0)

options = IdOptions(f_tol=1e-64, extended_trace=false, iterations = 100, autodiff=false, show_trace=true, estimate_initial=false)
@time sys = pem(data2, model, x0+0.05*randn(m))
sys.F
sys.B
sys.info.opt

sys = stmcb(data2,model)
_stmcb(data2, model, options)

options.OptimizationOptions.iterations


b = [0.3]
f = [0.5]

B = [0;b]
F = [1;f]

nb = nf = 10

N  = 200
u1 = 1*randn(N)
u2 = 1*randn(N)
lambda = 0.1
e1 = sqrt(lambda)*randn(N)
e2 = sqrt(lambda)*randn(N)
y1 = filt(B,F,u1) + 0.*filt(B,F,u2) + e1
y2 = 0.1*filt(B,F,u1) + filt(B,F,u2) + e2

u = hcat(u1,u2)
y = hcat(y1)
data2 = iddata(y, u)

ny = size(y,2)
nu = size(u,2)


mna = ones(Int,ny,ny)
mnb = 5*ones(Int,ny,nu)
mnf = 5*ones(Int,ny,nu)
mnc = 2*ones(Int,ny)
mnd = 2*ones(Int,ny)
mnk = ones(Int,ny,nu)
order = MPolyOrder(mna,mnb,mnf,mnc,mnd,mnk)
model = PolyModel(order, ny, nu, ControlCore.Siso{false}, CUSTOM)
nparam = sum(mna) + sum(mnb) + sum(mnf) + sum(mnc) + sum(mnd)
x = zeros(nparam)
x[1] = x[5] = 0.2
x[9] = x[13] = 0.4

x
sum(mnf)

predict(data2,model,x)
cost(data2,model,x)

options2 = IdOptions(f_tol=1e-64, extended_trace=false, iterations = 10, autodiff=true, show_trace=true, estimate_initial=false)
@time sys2 = pem(data2,model,x,options2)
sys2.info.opt
sys2.info.mse
sys2.B
sys2.F
