println("Starting ARX method test...")
# TODO: Are we exposing IdMFD to the user? If yes, we should `export` it.
#       Otherwise, we should be `using` it here, when needed.
using IdentificationToolbox: IdMFD

# Define the true system
a = [0.5, 0.2, 0.1]
b = [0.1, 0.3, 0.2]

# model orders
na, nb = 30, 30
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
S = Array(IdMFD,0)

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


u1 = randn(N)
u2 = randn(N)
lambda = 0.1
e1 = sqrt(lambda)*randn(N)
e2 = sqrt(lambda)*randn(N)
y1 = filt(B,A,u1) + filt(B,A,u2) + filt(1,A,e1)
y2 = filt(B,A,u1) + filt(B,A,u2) + filt(1,A,e2)

data2 = iddata(hcat(y1), hcat(u1,u2))

push!(S, arx(data2, na, nb, [1, 1]))

model = ARX(na, nb, [1, 1], 1, 2)
@time _arx(data2, model)

ny = 1
nu = 2
@time sys = pem(data2, model, randn((na*ny^2+nb*ny*nu)), options=IdOptions(autodiff=true,estimate_initial=false))

sys.info.mse
sys.info.opt
