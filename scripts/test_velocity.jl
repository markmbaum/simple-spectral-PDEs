using DrWatson
@quickactivate "NPDE"

using FastSphericalHarmonics
using BasicInterpolators
using ForwardDiff: derivative
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

lm = [(2,-2), (2,0), (4,-4)]

lmax = 24
θ, ϕ = sph_points(lmax+1)
θ, ϕ = ones(length(ϕ))' .* θ, ϕ' .* ones(length(θ))
ζₛ = zeros(lmax+1, 2lmax+1)
for (l,m) ∈ lm
    ζₛ[sph_mode(l,m)] = 1
end
#for l ∈ 1:lmax, m ∈ -l:l
#    ζₛ[sph_mode(l,m)] = randn()/l^2.5
#end
ζ = ζₛ |> evaluate
figure()
lim = ζ .|> abs |> maximum
pcolormesh(ϕ, θ, ζ, cmap="RdBu", vmin=-lim, vmax=lim)#, shading="gouraud")
colorbar()

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

quiver(ϕ, θ, u, v, pivot="middle")

##

uitp = BicubicInterpolator(ϕ[1,:], θ[:,1], (u .* sin.(θ))')
vitp = BicubicInterpolator(ϕ[1,:], θ[:,1], v')

∂u = zeros(size(θ))
∂v = zeros(size(θ))
for i ∈ 1:size(θ,1), j ∈ 1:size(ϕ,2)
    ∂u[i,j] = derivative(θ -> uitp(ϕ[i,j], θ), θ[i,j])
    ∂v[i,j] = derivative(ϕ -> vitp(ϕ, θ[i,j]), ϕ[i,j])
end

figure()
vort = @. ∂v/sin(θ) - ∂u/sin(θ)
lim = vort .|> abs |> maximum
pcolormesh(ϕ, θ, vort, cmap="RdBu", vmin=-lim, vmax=lim)
colorbar()

figure()
resid = vort .- ζ
lim = resid .|> abs |> maximum
pcolormesh(ϕ, θ, resid, cmap="RdBu", vmin=-lim, vmax=lim)
colorbar()