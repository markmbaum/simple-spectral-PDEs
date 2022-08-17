using DrWatson
@quickactivate "Simple Spectral PDEs"
push!(LOAD_PATH, srcdir())
using SimpleSpectralPDEs

##

using PyPlot
pygui(true)

##

model = KortewegDeVries()
x = gridpoints(model.N)
u₀ = 1e2*randominit(model)
tspan = [0, 10]

##

sol = integrate(model, u₀, tspan)

figure(constrained_layout=true)
t = LinRange(tspan[1], tspan[end], 101)
lim = sol .|> abs |> maximum
r = pcolormesh(
    x,
    t, 
    sol.(t),
    cmap="RdBu",
    vmin=-lim,
    vmax=lim,
    shading="gouraud"
)
xlabel("x")
ylabel("t")
cb = colorbar()
#cb.set_label("u", rotation=0);

figure()
t = LinRange(tspan[1], tspan[end], 20)
for i in 1:length(t)
    plot(x, sol(t[i]))
end

##

