println("Starting MORSM method test...")

# Define the true system
b = [0.1, 0.3, 0.2]
c = [0.5, 0.08]
d = [0.8, 0.1]
f = [0.5, 0.2, 0.1]

# model orders
nb, nc, nd, nf = 3, 1, 1, 3
nk = 1
n = [nb, nf, nk]

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

# test user methods
S = Array(IdDSisoRational,0)

# morsm constructor
push!(S, morsm(data, n))
push!(S, morsm(data, n,
  MORSM(ic=:zero, filter=:data, version=:G, loop=:open,
    nbrorders=3, maxiter=4, tol=1e-8)))

push!(S, morsm(data, [nb;nc;nd;nf;nk], MORSM(version=:H)))

# pem constructor
push!(S, pem(data, n, MORSM()))

push!(S, pem(data, [nb;nc;nd;nf;nk], BJ(), MORSM(version=:H)))
for system in S
  @test abs(system.info.mse-lambda) < 0.3*lambda
end
