using DrWatson
@quickactivate "NPDE"
push!(LOAD_PATH, srcdir())
using PDE1D

using OrdinaryDiffEq
using PyPlot

pygui(true)

##

model = AdvectionDiffusion((x,t) -> 2 + cos(x), D=0.01)
x = gridpoints(model.N)
u₀ = 1e2*randominit(model)
tspan = [0, 25]

prob = ODEProblem(∂u!, u₀, tspan, model);

##

sol = solve(prob, FBDF(autodiff=false), reltol=1e-6)

figure(constrained_layout=true)
t = LinRange(tspan[1], tspan[end], 101)
lim = sol .|> abs |> maximum
r = pcolormesh(
    x/(2π),
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

