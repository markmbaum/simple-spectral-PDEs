module SimpleSpectralPDEs

using FFTW
using UnPack
using OrdinaryDiffEq

export evaluate_terms!, โu!

#------------------------------------------------------------------------------
# general

export fourier_derivative!, gridpoints, randominit

function fourier_derivative!(F::Vector{๐ฏ})::Nothing where {๐ฏ<:Complex}
    N = length(F)
    @assert ispow2(N)
    @inbounds for k โ 1:Nรท2
        F[k] *= im*(k-1)
    end
    @inbounds for k โ Nรท2+1:N
        F[k] *= im*(k-1-N)
    end
    nothing
end

function fourier_derivative!(out, โ, stage, Pแตข, nderiv::Int)::Nothing
    @assert length(out) == length(โ) == length(stage)
    #evaluate derivatives in fourier space
    for _ โ 1:nderiv
        fourier_derivative!(โ)
    end
    #copy over before in-place ifft
    copyto!(stage, โ)
    Pแตข * stage
    #shuffle derivatives into output array
    out .= real.(stage)

    nothing
end

function checksetup(N, ๐ฏ)::Nothing
    @assert ispow2(N)
    @assert ๐ฏ <: AbstractFloat
    nothing
end

function gridpoints(N::Int, ๐ฏ=Float64)
    x = LinRange(0, 2ฯ, N+1)[1:end-1]
    x = collect(๐ฏ, x)
    return x
end

function randominit(model::๐ฏ, nmax::Int=8) where {๐ฏ}
    @unpack N, Pแตข = model
    ๐ฐ = fieldtype(๐ฏ, 1) |> eltype
    #load random values into a spectral vector
    @assert nmax + 1 โค N
    y = zeros(Complex{๐ฐ}, N)
    for n โ 2:nmax+1
        y[n] = randn(Complex{๐ฐ})/exp2(โn)
    end
    #take the ifft and return reals
    Pแตข * y
    return y .|> real
end

#------------------------------------------------------------------------------
# simple advection-diffusion Equation

export AdvectionDiffusion

struct AdvectionDiffusion{๐ฏ, ๐ฐ, ๐ฑ, ๐ฒ}
    x::Vector{๐ฏ}
    F::Vector{Complex{๐ฏ}}
    โ::Vector{Complex{๐ฏ}} #staging vector for fourier derivatives
    uโ::Vector{๐ฏ}
    uโโ::Vector{๐ฏ}
    D::๐ฏ
    N::Int64
    P::๐ฐ #fft plan
    Pแตข::๐ฑ #ifft plan
    ๐::๐ฒ #velocity function ๐(x,t)
end

function Base.show(io::IO, model::AdvectionDiffusion{๐ฏ}) where {๐ฏ}
    println(io, "$(model.N) point AdvectionDiffusion{$๐ฏ}")
end

AdvectionDiffusion(; kw...) = AdvectionDiffusion((x,t) -> 1.0; kw...)

AdvectionDiffusion(๐::Real; kw...) = AdvectionDiffusion((x,t) -> ๐; kw...)

function AdvectionDiffusion(๐::Function; D=0.0, N::Int=128, ๐ฏ::Type=Float64) # ๐(x,t)
    checksetup(N, ๐ฏ)
    x = gridpoints(N, ๐ฏ)
    F = zeros(Complex{๐ฏ}, N)
    โ = zeros(Complex{๐ฏ}, N)
    uโ = zeros(๐ฏ, N)
    uโโ = zeros(๐ฏ, N)
    P = plan_fft!(F, flags=FFTW.PATIENT)
    Pแตข = plan_ifft!(F, flags=FFTW.PATIENT)
    AdvectionDiffusion(x, F, โ, uโ, uโโ, convert(๐ฏ,D), N, P, Pแตข, ๐)
end

function evaluate_terms!(model::AdvectionDiffusion{๐ฏ}, u::AbstractVector{๐ฏ}) where {๐ฏ}
    #unpack model arrays
    @unpack F, โ, uโ, uโโ, N, P, Pแตข = model
    #check length
    @assert length(u) == N
    #copy u values into F for in-place DFT
    copyto!(F, u)
    #take the DFT in-place
    P * F
    #shuffle over for taking derivatives
    copyto!(โ, F)
    #first derivative
    fourier_derivative!(uโ, โ, F, Pแตข, 1)
    #second derivative
    fourier_derivative!(uโโ, โ, F, Pแตข, 1)
    return nothing
end

advection_diffusion(x, t, uโ, uโโ, D, ๐::โฑ) where โฑ = -๐(x,t)*uโ + D*uโโ

function โu!(โu, u, model::AdvectionDiffusion, t)::Nothing
    @unpack x, uโ, uโโ, D, ๐ = model
    evaluate_terms!(model, u)
    @inbounds for i โ eachindex(โu)
        โu[i] = advection_diffusion.(x[i], t, uโ[i], uโโ[i], D, ๐)
    end
    nothing
end

#------------------------------------------------------------------------------
# Korteweg-De Vries model

export KortewegDeVries

struct KortewegDeVries{๐ฏ, ๐ฐ, ๐ฑ}
    a::๐ฏ
    F::Vector{Complex{๐ฏ}}
    โ::Vector{Complex{๐ฏ}} #staging vector for fourier derivatives
    uโ::Vector{๐ฏ}
    uโโ::Vector{๐ฏ}
    uโโโ::Vector{๐ฏ}
    D::๐ฏ
    N::Int64
    P::๐ฐ #fft plan
    Pแตข::๐ฑ #ifft plan
end

function Base.show(io::IO, model::KortewegDeVries{๐ฏ}) where {๐ฏ}
    println(io, "$(model.N) point KortewegDeVries{$๐ฏ} with a=$(model.a)")
end

function KortewegDeVries(; a=0.1, N::Int=128, D::Real=0., ๐ฏ::Type=Float64)
    checksetup(N, ๐ฏ)
    F = zeros(Complex{๐ฏ}, N)
    โ = zeros(Complex{๐ฏ}, N)
    uโ = zeros(๐ฏ, N)
    uโโ = zeros(๐ฏ, N)
    uโโโ = zeros(๐ฏ, N)
    P = plan_fft!(F, flags=FFTW.PATIENT)
    Pแตข = plan_ifft!(F, flags=FFTW.PATIENT)
    KortewegDeVries(convert(๐ฏ, a), F, โ, uโ, uโโ, uโโโ, convert(๐ฏ, D), N, P, Pแตข)
end

function evaluate_terms!(model::KortewegDeVries{๐ฏ}, u::AbstractVector{๐ฏ}) where {๐ฏ}
    #unpack model arrays
    @unpack F, โ, uโ, uโโ, uโโโ, D, N, P, Pแตข = model
    #check length
    @assert length(u) == N
    #copy u values into F for in-place DFT
    copyto!(F, u)
    #take the DFT in-place
    P * F
    #shuffle over for taking derivatives
    copyto!(โ, F)
    #first derivative
    fourier_derivative!(uโ, โ, F, Pแตข, 1)
    if D != zero(๐ฏ)
        #second derivative
        fourier_derivative!(uโโ, โ, F, Pแตข, 1)
        #third derivative
        fourier_derivative!(uโโโ, โ, F, Pแตข, 1)
    else
        #third derivative
        fourier_derivative!(uโโโ, โ, F, Pแตข, 2)
    end
    return nothing
end

korteweg_de_vries(u, uโ, uโโโ, a, uโโ, D) = -u*uโ - a*a*uโโโ + D*uโโ

function โu!(โu, u, model::KortewegDeVries, t)::Nothing
    @unpack uโ, uโโ, uโโโ, a, D = model
    evaluate_terms!(model, u)
    โu .= korteweg_de_vries.(u, uโ, uโโโ, a, uโโ, D)
    nothing
end

#------------------------------------------------------------------------------
# Kuramoto-Sivashinsky model

export KuramotoSivashinsky

struct KuramotoSivashinsky{๐ฏ, ๐ฐ, ๐ฑ}
    L::๐ฏ
    F::Vector{Complex{๐ฏ}}
    โ::Vector{Complex{๐ฏ}} #staging vector for fourier derivatives
    uโ::Vector{๐ฏ}
    uโโ::Vector{๐ฏ}
    uโโโโ::Vector{๐ฏ}
    N::Int64
    P::๐ฐ #fft plan
    Pแตข::๐ฑ #ifft plan
end

function Base.show(io::IO, model::KuramotoSivashinsky{๐ฏ}) where {๐ฏ}
    println(io, "$(model.N) point KuramotoSivashinsky{$๐ฏ} with L=$(model.L)")
end

function KuramotoSivashinsky(; L=2.5, N::Int=128, ๐ฏ::Type=Float64)
    checksetup(N, ๐ฏ)
    F = zeros(Complex{๐ฏ}, N)
    โ = zeros(Complex{๐ฏ}, N)
    uโ = zeros(๐ฏ, N)
    uโโ = zeros(๐ฏ, N)
    uโโโโ = zeros(๐ฏ, N)
    P = plan_fft!(F, flags=FFTW.PATIENT)
    Pแตข = plan_ifft!(F, flags=FFTW.PATIENT)
    KuramotoSivashinsky(convert(๐ฏ, L), F, โ, uโ, uโโ, uโโโโ, N, P, Pแตข)
end

function evaluate_terms!(model::KuramotoSivashinsky{๐ฏ}, u::AbstractVector{๐ฏ}) where {๐ฏ}
    #unpack model arrays
    @unpack L, F, โ, uโ, uโโ, uโโโโ, N, P, Pแตข = model
    #check length
    @assert length(u) == N
    #copy u values into F for in-place DFT
    copyto!(F, u)
    #take the DFT in-place
    P * F
    #shuffle over for taking derivatives
    copyto!(โ, F)
    #first derivative
    fourier_derivative!(uโ, โ, F, Pแตข, 1)
    uโ ./= L
    #second derivative
    fourier_derivative!(uโโ, โ, F, Pแตข, 1)
    uโโ ./= L^2
    #fourth derivative
    fourier_derivative!(uโโโโ, โ, F, Pแตข, 2)
    uโโโโ ./= L^4

    return nothing
end

kuramoto_sivashinsky(u, uโ, uโโ, uโโโโ) = -(uโโ + uโโโโ + u*uโ)

function โu!(โu, u, model::KuramotoSivashinsky, t)::Nothing
    @unpack uโ, uโโ, uโโโโ = model
    evaluate_terms!(model, u)
    โu .= kuramoto_sivashinsky.(u, uโ, uโโ, uโโโโ)
    nothing
end

#------------------------------------------------------------------------------
# integration

export integrate

#convenience/barrier
function integrate(model, uโ, tspan; method::Symbol=:QNDF, tol=1e-9)
    integrate(model, uโ, tspan, method |> eval, tol)
end

function integrate(model, uโ, tspan, method, tol)
    solve(
        ODEProblem(โu!, uโ, tspan, model),
        method(autodiff=false),
        reltol=tol
    )
end

end