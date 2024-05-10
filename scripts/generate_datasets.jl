using Pkg
Pkg.activate(@__DIR__)
using SimpleSpectralPDEs
using NetCDF

##

using PyPlot
pygui(true)

##

function datadir(args...)
    joinpath(@__DIR__, "..", "data", args...)
end

##

function generate_dataset(model, tspan, nt, fnout)

    u₀ = 1e2 * randominit(model)
    sol = integrate(model, u₀, tspan)

    t = LinRange(tspan[1], tspan[end], nt)
    x = gridpoints(model.N)
    u = zeros(Float32, length(x), length(t))
    for i ∈ eachindex(t)
        u[:, i] = sol(t[i])
    end

    if isfile(fnout)
        rm(fnout)
    end
    nccreate(
        fnout,
        "u",
        "x",
        x .|> Float32,
        Dict("name" => "spatial coordinate"),
        "t",
        t .|> Float32,
        Dict("name" => "time"),
        t=NC_FLOAT
    )
    ncwrite(u, fnout, "u")
    ncclose(fnout)

    figure(constrained_layout=true)
    lim = u .|> abs |> maximum
    r = pcolormesh(x[1:2:end], t[1:4:end], u[1:2:end, 1:4:end]', cmap="RdBu", vmin=-lim, vmax=lim, shading="gouraud")
    xlabel("x")
    ylabel("t")
    colorbar()

    return nothing
end

##

generate_dataset(
    AdvectionDiffusion((x, t) -> 1.25 + cos(x), D=0, N=128),
    [0, 50],
    5_000,
    datadir("sims", "advection.nc")
)

##

generate_dataset(
    KortewegDeVries(N=128, D=1e-8),
    [0, 200],
    20_000,
    datadir("sims", "korteweg_de_vries.nc")
)

##

generate_dataset(
    KuramotoSivashinsky(N=128, L=10),
    [0, 5000],
    500_000,
    datadir("sims", "kuramoto_sivashinsky.nc")
)