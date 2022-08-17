using DrWatson
@quickactivate "Simple Spectral PDEs"
push!(LOAD_PATH, srcdir())
using SimpleSpectralPDEs

using GLMakie

##

model = KuramotoSivashinsky(N=256, L=6)
x = gridpoints(model.N)
u₀ = 1e2*randominit(model, 8) #@. cos(x)*sin(3x) #
tspan = [0, 100];

##

sol = integrate(model, u₀, tspan);

##

time = Observable(0.0)
y = @lift(sol($time))
yₑ = @lift(sol($time)[end])

fig = lines(x, y, linewidth=4)
fig.axis.xlabel = "x"
fig.axis.ylabel = "u"
ylims!(-3, 3)
framerate = 60
times = range(0, tspan[end], step=20/framerate)

record(fig, plotsdir("animation.gif"), times; framerate=framerate) do t
    time[] = t
end

##

