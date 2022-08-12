using DrWatson
@quickactivate "NPDE"
push!(LOAD_PATH, srcdir())
using NPDE

using OrdinaryDiffEq
using PyPlot

pygui(true)

##

model = KuramotoSivashinsky(N=256, L=15)
x = gridpoints(model.N)
u₀ = 1e2*randominit(model) #@. cos(2x)*sin(3x)
tspan = [0, 500]

prob = ODEProblem(∂u!, u₀, tspan, model);

##

sol = solve(prob, FBDF(autodiff=false), reltol=1e-12)

figure(constrained_layout=true)
t = LinRange(tspan[1], tspan[end], 501)
lim = sol .|> abs |> maximum
r = pcolormesh(
    x*model.L/(2π), 
    t, 
    sol.(t),
    cmap="RdBu",
    vmin=-lim,
    vmax=lim,
    shading="gouraud"
)
xlabel("x")
ylabel("t")
#cb = colorbar()
#cb.set_label("u", rotation=0);

##

figure()
t = LinRange(tspan[1], 25, 20)
for i in 1:length(t)
    plot(x, sol(t[i]))
end

##

