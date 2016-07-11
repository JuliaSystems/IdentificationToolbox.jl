println("Starting FIR method test...")

# Define the true system
b = [0.7, 0.3, 0.2]

# model orders
nb = 3
nk = 1
n  = [nb, nk]

# intitial parameters
x = [b; zeros(nb-3)]

# generate input data+noise and simulate output
B = [0;b]
N = 1000
u = randn(N)
λ = 10
e = sqrt(λ)*randn(N)
y = filt(B,1,u) + e

# create iddataObject for the input/output data
data = iddata(y,u)

# test user methods
S = Array(IdDSisoRational,0)

# arx constructor
push!(S, fir(data, n))
push!(S, fir(data, n, FIR()))
push!(S, fir(data, n, FIR(ic = :truncate)))
push!(S, fir(data, n, FIR(ic = :zero)))

# pem constructor
push!(S, pem(data, n, FIR()))
push!(S, pem(data, n, FIR(ic = :truncate)))
push!(S, pem(data, n, FIR(ic = :zero)))

for system in S
  @test abs(system.info.mse-lambda) < 0.3*lambda
  showall(system.G)
end
