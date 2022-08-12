using DrWatson
@quickactivate "NPDE"
using NPDE

using OrdinaryDiffEq
using FastSphericalHarmonics
using PyPlot
pygui(true)

##

trim(x) = abs2(x) < 100eps(x) ? zero(x) : x

transform(x) = x |> sph_transform .|> trim

evaluate(x) = x |> sph_evaluate .|> trim

fϵ(l, m) = sqrt((l^2 - m^2)/(4l^2 - 1))

function colatscale(x)
    θ, _ = sph_points(size(x,1))
    y = similar(x)
    for i ∈ 1:length(θ)
        y[i,:] .= x[i,:]/sin(θ[i])
    end
    return y
end

##

function vorticity_equation(ζ, args...)
    
    ζₛ = ζ |> transform

    vₛ = zeros(size(ζₛ))
    for l ∈ 1:lmax, m ∈ -l:l
        vₛ[sph_mode(l,-m)] = m*ζₛ[sph_mode(l,m)]/(l*(l+1))
    end
    v = vₛ |> evaluate |> colatscale

    uₛ = zeros(size(ζₛ))
    for l ∈ 0:lmax-1
        uₛ[sph_mode(l,-l)] = -fϵ(l+1,-l)*ζₛ[sph_mode(l+1,-l)]/(l+1)
        for m ∈ -l+1:l-1
            uₛ[sph_mode(l,m)] = -fϵ(l+1,m)*ζₛ[sph_mode(l+1,m)]/(l+1) + fϵ(l,m)*ζₛ[sph_mode(l-1,m)]/l
        end
        uₛ[sph_mode(l,l)] = -fϵ(l+1,l)*ζₛ[sph_mode(l+1,l)]/(l+1)
    end
    for m ∈ -lmax+1:lmax-1
        uₛ[sph_mode(lmax,m)] = fϵ(lmax,m)*ζₛ[sph_mode(lmax-1,m)]/lmax
    end
    u = uₛ |> evaluate |> colatscale

    dxₛ = zeros(size(ζₛ))
    for l ∈ 1:lmax, m ∈ -l:l
        dxₛ[sph_mode(l,-m)] = -m*ζₛ[sph_mode(l,m)]
    end
    dx = dxₛ |> evaluate |> colatscale

    dyₛ = zeros(size(ζₛ))
    for l ∈ 0:lmax-1
        dyₛ[sph_mode(l,-l)] = -(l+2)*fϵ(l+1,-l)*ζₛ[sph_mode(l+1,-l)]
        for m ∈ -l+1:l-1
            dyₛ[sph_mode(l,m)] = -(l+2)*fϵ(l+1,m)*ζₛ[sph_mode(l+1,m)] + (l-1)*fϵ(l,m)*ζₛ[sph_mode(l-1,m)]
        end
        dyₛ[sph_mode(l,l)] = -(l+2)*fϵ(l+1,l)*ζₛ[sph_mode(l+1,l)]
    end
    for m ∈ -lmax+1:lmax-1
        dyₛ[sph_mode(lmax,m)] = (lmax-1)*fϵ(lmax,m)*ζₛ[sph_mode(lmax-1,m)]
    end
    dy = dyₛ |> evaluate |> colatscale

    @. -u*dx - v*dy
end

##

lm = [(2,0), (10,-9)]

lmax = 64
θ, ϕ = sph_points(lmax+1)
θ, ϕ = ones(length(ϕ))' .* θ, ϕ' .* ones(length(θ))
ζₛ = zeros(lmax+1, 2lmax+1)
for (l,m) ∈ lm
    ζₛ[sph_mode(l,m)] = 1
end
ζ = ζₛ |> evaluate
figure()
lim = ζ .|> abs |> maximum
pcolormesh(ϕ, θ, ζ, cmap="RdBu", vmin=-lim, vmax=lim, shading="gouraud")
colorbar()

##

dζ = vorticity_equation(ζ)
figure()
lim = dζ .|> abs |> maximum
pcolormesh(dζ, vmin=-lim, vmax=lim, cmap="RdBu")
colorbar()

##

Z = zeros(16, size(ζ)...)
Z[1,:,:] .= ζ
Δt = 4e-1
for i ∈ 2:size(Z,1)
    Z[i,:,:] .= Δt*vorticity_equation(Z[i-1,:,:]) .+ Z[i-1,:,:]
end

fig, axs = subplots(4, 4, constrained_layout=true, figsize=(10,6))
for i ∈ 1:4, j ∈ 1:4
    idx = 4*(i-1) + j
    z = Z[idx,:,:]
    lim = z .|> abs |> maximum
    axs[i,j].pcolormesh(ϕ, θ, z, vmin=-lim, vmax=lim, cmap="RdBu")
    axs[i,j].set_xticks([])
    axs[i,j].set_yticks([])
    axs[i,j].set_title("time $idx")
end

##