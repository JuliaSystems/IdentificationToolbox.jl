println("Starting BJ method test...")

# Define the true system
b = [0.1, 0.3, 0.2]
c = [0.5, 0.11, 0.05]
d = [0.2, 0.1, 0.02]
f = [0.5, 0.2, 0.1]

# model orders
nb, nc, nd, nf = 30, 30, 30, 30
nk = 1
n = [nb, nc, nd, nf, nk]

# intitial parameters
x = [b; zeros(nb-3);c; zeros(nc-3); d; zeros(nd-3); f; zeros(nf-3)]

# generate input data+noise and simulate output
B = [0;b]
C = [1;c]
D = [1;d]
F = [1;f]
N = 1000
u = randn(N)
lambda = 10
e = sqrt(lambda)*randn(N)
y = filt(B,F,u,[2.;2;2]) + filt(C,D,e)

# create iddataObject for the input/output data
data = iddata(y,u)

# test internal method
k = nb+nc+nd+nf
storage = zeros(Float64,k,k+1)
last_V = [-0.1]

V = IdentificationToolbox.gradhessian!(data, n, x, BJ(), 0.9*x, last_V, storage)
@assert abs(V-lambda) < 0.3*lambda

# test user methods
S = Array(IdDSisoRational,0)

# bj constructor
push!(S, bj(data, n, x))

# pem constructor
push!(S, pem(data, n, x, BJ()))

for system in S
  @test abs(system.info.mse-lambda) < 0.3*lambda
end
