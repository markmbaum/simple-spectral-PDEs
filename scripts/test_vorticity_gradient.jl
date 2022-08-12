using DrWatson
@quickactivate "NPDE"
using NPDE

##

using FastSphericalHarmonics
using BasicInterpolators
using ForwardDiff: gradient
using PyPlot
pygui(true)

##

trim(x) = abs2(x) < 100eps(x) ? zero(x) : x;

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

lm = [(3,0), (4,-3)]


lmax = 256
θ, ϕ = sph_points(lmax+1)
θ, ϕ = ones(length(ϕ))' .* θ, ϕ' .* ones(length(θ))
ζₛ = zeros(lmax+1, 2lmax+1)
#for (l,m) ∈ lm
#    ζₛ[sph_mode(l,m)] = 1
#end
for l ∈ 1:10, m ∈ -l:l
    ζₛ[sph_mode(l,m)] = randn()/l^2.5
end
ζ = ζₛ |> evaluate
figure()
lim = ζ .|> abs |> maximum
pcolormesh(ϕ, θ, ζ, cmap="RdBu", vmin=-lim, vmax=lim, shading="gouraud")
colorbar()

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

#quiver(ϕ, θ, dx, dy, pivot="middle")

##

itp = BicubicInterpolator(ϕ[1,:], θ[:,1], ζ')

ix = zeros(size(θ))
iy = zeros(size(θ))
for i ∈ 1:size(θ,1), j ∈ 1:size(ϕ,2)
    iy[i,j], ix[i,j] = gradient(z -> itp(z[2], z[1]), [θ[i,j], ϕ[i,j]])
end
ix ./= sin.(θ)

figure()
lim = ζ .|> abs |> maximum
pcolormesh(ϕ, θ, ζ, cmap="RdBu", vmin=-lim, vmax=lim, shading="gouraud")
colorbar()
#quiver(ϕ, θ, ix ./ θ, iy, pivot="middle")

figure()
resid = ix .- dx
lim = resid .|> abs |> maximum
pcolormesh(ϕ, θ, resid, cmap="RdBu", vmin=-lim, vmax=lim)
colorbar()
title("dx residual")

figure()
resid = iy .- dy
lim = resid .|> abs |> maximum
pcolormesh(ϕ, θ, resid, cmap="RdBu", vmin=-lim, vmax=lim)
colorbar()
title("dy residual")