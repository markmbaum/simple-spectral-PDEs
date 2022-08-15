using DrWatson
@quickactivate "PDE1D"
push!(LOAD_PATH, srcdir())
using PDE1D

using GLMakie

##

model = AdvectionDiffusion((x,t) -> 1.25 + cos(x + 2π*t/25), N=512, D=1e-4)
x = gridpoints(model.N)
u₀ = 3e2*randominit(model, 16) #@. cos(x)*sin(3x) #
tspan = [0, 100];

##

sol = integrate(model, u₀, tspan);

##

time = Observable(0.0)
y = @lift(sol($time))
yₑ = @lift(sol($time)[end])

fig = lines(x, y, linewidth=4)
ylims!(-1, 1)
framerate = 30
times = range(0, tspan[end], step=6/framerate)

record(fig, plotsdir("animation.mp4"), times; framerate=framerate) do t
    time[] = t
end

##

