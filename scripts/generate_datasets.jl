using DrWatson
@quickactivate "Simple Spectral PDEs"
push!(LOAD_PATH, srcdir())
using SimpleSpectralPDEs
using NetCDF

##

using PyPlot
pygui(true)

##

function generate_dataset(model, tspan, fnout)

    u₀ = 1e2*randominit(model)
    sol = integrate(model, u₀, tspan)

    t = LinRange(tspan[1], tspan[end], 10000)
    x = gridpoints(model.N)
    u = zeros(Float32, length(x), length(t))
    for i ∈ eachindex(t)
        u[:,i] = sol(t[i])
    end

    if isfile(fnout)
        rm(fnout)
    end
    nccreate(
        fnout,
        "u",
        "x",
        x .|> Float32,
        Dict("name"=>"spatial coordinate"),
        "t",
        t .|> Float32,
        Dict("name"=>"time"),
        t=NC_FLOAT
    )
    ncwrite(u, fnout, "u")
    ncclose(fnout)
    
    figure(constrained_layout=true)
    lim = u .|> abs |> maximum
    r = pcolormesh(x[1:3:end], t[1:3:end], u[1:3:end,1:3:end]', cmap="RdBu", vmin=-lim, vmax=lim, shading="gouraud")
    xlabel("x")
    ylabel("t")
    cb = colorbar()

    return nothing
end

##

generate_dataset(
    AdvectionDiffusion((x,t) -> 1.25 + cos(x), D=0, N=256),
    [0, 20],
    datadir("sims", "advection.nc")
)

##

generate_dataset(
    KortewegDeVries(N=256),
    [0, 30],
    datadir("sims", "korteweg_de_vries.nc")
)

##

generate_dataset(
    KuramotoSivashinsky(N=256, L=10),
    [0, 2000],
    datadir("sims", "kuramoto_sivashinsky.nc")
)