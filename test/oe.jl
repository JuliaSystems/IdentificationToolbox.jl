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


b = [0.1, 0.3, 0.2]
f = [0.5, 0.11, 0.05]

B = [0;b]
F = [1;f]

N = 1000
u1 = randn(N)
u2 = randn(N)
lambda = 0.1
e1 = sqrt(lambda)*randn(N)
e2 = sqrt(lambda)*randn(N)
y1 = filt(B,F,u1) + filt(B,F,u2) + e1
y2 = filt(B,F,u1) + filt(B,F,u2) + e2

u = hcat(u1,u2)
y = hcat(y1,y2)
data2 = iddata(y, u)

model = OE(2,2,[1,1])

ny = 2
nu = 2
@time y_est = predict(data2,model,zeros((2+2)*ny*nu))
cost(data2,model,zeros((2+2)*ny*nu))
y_est
sumabs2(y - y_est)

data2.y

model.orders.nf
reshape(randn(16),8,2)

@time begin
  Bm = vcat(zeros(2,2),randn(6,2))
  Fm = vcat(eye(2),randn(6,2))
  _filt_iir!(out, Bm, Fm, u, rand(2,2))
end

@time begin
  Pb = PolyMatrix(vcat(zeros(2,2),randn(6,2)), (2,2))
  Pf = PolyMatrix(vcat(eye(2),randn(6,2)),(2,2))
  _filt_iir!(out, Pb, Pf, u, rand(2,2))
end

_zerosi(Pb,Pf,Float64)

out = zeros(N,2)
_filt_fir!(out, Pb, u, rand(2,2))
out


Array(Float64, size(Pf,1),size(Pb,2))

@which filt(Pb,Pf,randn(100,2))

insert!(Pf, 4, randn(2,2))

Pf
filt(Pb,Pf,randn(100,2))

size(Pf,2)
