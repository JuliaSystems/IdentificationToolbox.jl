println("Starting ARMAX method test...")

# Define the true system
a = [0.5, 0.2, 0.1]
b = [0.1, 0.3, 0.2]
c = [0.5, 0.11, 0.05]

# model orders
na, nb, nc = 30, 30, 30
nk = 1
n = [na, nb, nc, nk]

# intitial parameters
x = [a; zeros(na-3);b; zeros(nb-3);c; zeros(nc-3)]

# generate input data+noise and simulate output
A = [1;a]
B = [0;b]
C = [1;c]
N = 1000
u = randn(N)
lambda = 10
e = sqrt(lambda)*randn(N)
y = filt(B,A,u) + filt(C,A,e)

# create iddataObject for the input/output data
data = iddata(y,u)

# test internal method
k = na+nb+nc
storage = zeros(Float64,k,k+1)
last_V = [-0.1]
V = IdentificationToolbox.gradhessian!(data, n, x, ARMAX(), 0.9*x, last_V, storage)
@assert abs(V-lambda) < 0.1*lambda

# test user methods
S = Array(IdDSisoRational,0)

# armax constructor
push!(S, armax(data, n..., x))

# pem constructor
push!(S, pem(data, n, x, ARMAX()))

for system in S
  @test abs(system.info.mse-lambda) < 0.3*lambda
end
