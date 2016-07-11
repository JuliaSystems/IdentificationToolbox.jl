println("Starting Steiglitz-McBride method test...")

# Define the true system
b = [0.3, 0.1]
f = [0.5, 0.23]

# model orders
nb, nf = 2, 2
nk = 1
n = [nb, nf, nk]

# generate input data+noise and simulate output
B = [0.0;b]
F = [1.0;f]
N = 1000
u = randn(N)
lambda = 0.05
e = sqrt(lambda)*randn(N)
y = filt(B,F,u) + e

# create iddataObject for the input/output data
data = iddata(y,u,0.1)

# test user methods
S = Array(IdDSisoRational,0)

# bj constructor
push!(S, stmcb(data, n))

# pem constructor
push!(S, pem(data, n, STMCB()))

# test feedthrough (if used to identify e.g. a noise model)
C = [1;b]
D = [1;f]
u2 = randn(N)
e2 = sqrt(lambda)*randn(N)
y2 = filt(C,D,u2) + e2

# identify with feedthrough
data2 = iddata(y2, u2, 0.1)
push!(S, pem(data2, n, STMCB(feedthrough=true)))

for system in S
  abs(system.info.mse-lambda) < lambda
end
