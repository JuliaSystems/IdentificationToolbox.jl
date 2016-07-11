println("Starting ARX method test...")

# Define the true system
a = [0.5, 0.2, 0.1]
b = [0.1, 0.3, 0.2]

# model orders
na, nb = 3, 3
nk = 1
n = [na, nb, nk]

# intitial parameters
x = [a; zeros(na-3);b; zeros(nb-3)]

# generate input data+noise and simulate output
A = [1;a]
B = [0;b]
N = 1000
u = randn(N)
lambda = 10
e = sqrt(lambda)*randn(N)
y = filt(B,A,u) + filt(1,A,e)

# create iddataObject for the input/output data
data = iddata(y,u)

# test user methods
S = Array(IdDSisoRational,0)

# arx constructor
push!(S, arx(data, n))
push!(S, arx(data, n, ARX()))
push!(S, arx(data, n, ARX(ic = :truncate)))
push!(S, arx(data, n, ARX(ic = :zero)))

# pem constructor
push!(S, pem(data, n, ARX()))
push!(S, pem(data, n, ARX(ic = :truncate)))
push!(S, pem(data, n, ARX(ic = :zero)))

for system in S
  @test abs(system.info.mse-lambda) < 0.3*lambda
end
