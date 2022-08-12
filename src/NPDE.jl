module NPDE

using FFTW
using FastSphericalHarmonics
using UnPack

import Base.show

export evaluate_derivatives!, ∂u!

#------------------------------------------------------------------------------
# general

export fourier_derivative!, gridpoints, randominit

abstract type PDE1D{𝒯<:AbstractFloat, 𝒰, 𝒱} end

function fourier_derivative!(F::Vector{𝒯})::Nothing where {𝒯<:Complex}
    N = length(F)
    @assert ispow2(N)
    @inbounds for k ∈ 1:N÷2
        F[k] *= im*(k-1)
    end
    @inbounds for k ∈ N÷2+1:N
        F[k] *= im*(k-1-N)
    end
    nothing
end

function fourier_derivative!(out, ∂, stage, Pᵢ, nderiv::Int)::Nothing
    @assert length(out) == length(∂) == length(stage)
    #evaluate derivatives in fourier space
    for _ ∈ 1:nderiv
        fourier_derivative!(∂)
    end
    #copy over before in-place ifft
    copyto!(stage, ∂)
    Pᵢ * stage
    #shuffle derivatives into output array
    out .= real.(stage)

    nothing
end

function checksetup(N, 𝒯)::Nothing
    @assert ispow2(N)
    @assert 𝒯 <: AbstractFloat
    nothing
end

function gridpoints(N::Int, 𝒯=Float64)
    x = LinRange(0, 2π, N+1)[1:end-1]
    x = collect(𝒯, x)
    return x
end

function randominit(model::PDE1D{𝒯}) where {𝒯}
    @unpack F, N, Pᵢ = model
    #load random values into F
    F[1] = 0im
    for n ∈ 2:9
        F[n] = randn(Complex{𝒯})/(n-1)
    end
    #take the ifft
    Pᵢ * F
    return F .|> real
end

#------------------------------------------------------------------------------
# Korteweg-De Vries model

export KortewegDeVries

struct KortewegDeVries{𝒯, 𝒰, 𝒱} <: PDE1D{𝒯, 𝒰, 𝒱}
    a::𝒯
    F::Vector{Complex{𝒯}}
    ∂::Vector{Complex{𝒯}} #staging vector for fourier derivatives
    uₓ::Vector{𝒯}
    uₓₓₓ::Vector{𝒯}
    N::Int64
    P::𝒰 #fft plan
    Pᵢ::𝒱 #ifft plan
end

function Base.show(io::IO, model::KortewegDeVries{𝒯}) where {𝒯}
    println(io, "$(model.N) point KortewegDeVries{$𝒯} with a=$(model.a)")
end

function KortewegDeVries(; a=0.1, N::Int=128, 𝒯::Type=Float64)
    checksetup(N, 𝒯)
    F = zeros(Complex{𝒯}, N)
    ∂ = zeros(Complex{𝒯}, N)
    uₓ = zeros(𝒯, N)
    uₓₓₓ = zeros(𝒯, N)
    P = plan_fft!(F, flags=FFTW.PATIENT)
    Pᵢ = plan_ifft!(F, flags=FFTW.PATIENT)
    KortewegDeVries(convert(𝒯, a), F, ∂, uₓ, uₓₓₓ, N, P, Pᵢ)
end

function evaluate_derivatives!(model::KortewegDeVries{𝒯}, u::AbstractVector{𝒯}) where {𝒯}
    #unpack model arrays
    @unpack F, ∂, uₓ, uₓₓₓ, N, P, Pᵢ = model
    #check length
    @assert length(u) == N
    #copy u values into F for in-place DFT
    copyto!(F, u)
    #take the DFT in-place
    P * F
    #shuffle over for taking derivatives
    copyto!(∂, F)
    #first derivative
    fourier_derivative!(uₓ, ∂, F, Pᵢ, 1)
    #third derivative
    fourier_derivative!(uₓₓₓ, ∂, F, Pᵢ, 2)
    return nothing
end

korteweg_de_vries(u, uₓ, uₓₓₓ, a) = -u*uₓ - a*a*uₓₓₓ

function ∂u!(∂u, u, model::KortewegDeVries, t)::Nothing
    @unpack uₓ, uₓₓₓ, a = model
    evaluate_derivatives!(model, u)
    ∂u .= korteweg_de_vries.(u, uₓ, uₓₓₓ, a)
    nothing
end

#------------------------------------------------------------------------------
# Kuramoto-Sivashinsky model

export KuramotoSivashinsky

struct KuramotoSivashinsky{𝒯, 𝒰, 𝒱} <: PDE1D{𝒯, 𝒰, 𝒱}
    L::𝒯
    F::Vector{Complex{𝒯}}
    ∂::Vector{Complex{𝒯}} #staging vector for fourier derivatives
    uₓ::Vector{𝒯}
    uₓₓ::Vector{𝒯}
    uₓₓₓₓ::Vector{𝒯}
    N::Int64
    P::𝒰 #fft plan
    Pᵢ::𝒱 #ifft plan
end

function Base.show(io::IO, model::KuramotoSivashinsky{𝒯}) where {𝒯}
    println(io, "$(model.N) point KuramotoSivashinsky{$𝒯} with L=$(model.L)")
end

function KuramotoSivashinsky(; L=1.0, N::Int=128, 𝒯::Type=Float64)
    checksetup(N, 𝒯)
    F = zeros(Complex{𝒯}, N)
    ∂ = zeros(Complex{𝒯}, N)
    uₓ = zeros(𝒯, N)
    uₓₓ = zeros(𝒯, N)
    uₓₓₓₓ = zeros(𝒯, N)
    P = plan_fft!(F, flags=FFTW.PATIENT)
    Pᵢ = plan_ifft!(F, flags=FFTW.PATIENT)
    KuramotoSivashinsky(convert(𝒯, L), F, ∂, uₓ, uₓₓ, uₓₓₓₓ, N, P, Pᵢ)
end

function evaluate_derivatives!(model::KuramotoSivashinsky{𝒯}, u::AbstractVector{𝒯}) where {𝒯}
    #unpack model arrays
    @unpack L, F, ∂, uₓ, uₓₓ, uₓₓₓₓ, N, P, Pᵢ = model
    #check length
    @assert length(u) == N
    #copy u values into F for in-place DFT
    copyto!(F, u)
    #take the DFT in-place
    P * F
    #shuffle over for taking derivatives
    copyto!(∂, F)
    #first derivative
    fourier_derivative!(uₓ, ∂, F, Pᵢ, 1)
    uₓ ./= L
    #second derivative
    fourier_derivative!(uₓₓ, ∂, F, Pᵢ, 1)
    uₓₓ ./= L^2
    #fourth derivative
    fourier_derivative!(uₓₓₓₓ, ∂, F, Pᵢ, 2)
    uₓₓₓₓ ./= L^4

    return nothing
end

kuramoto_sivashinsky(u, uₓ, uₓₓ, uₓₓₓₓ) = -(uₓₓ + uₓₓₓₓ + u*uₓ)

function ∂u!(∂u, u, model::KuramotoSivashinsky, t)::Nothing
    @unpack uₓ, uₓₓ, uₓₓₓₓ = model
    evaluate_derivatives!(model, u)
    ∂u .= kuramoto_sivashinsky.(u, uₓ, uₓₓ, uₓₓₓₓ)
    nothing
end

#------------------------------------------------------------------------------
# the 2D Barotropic Vorticity equation on the sphere

#struct BarotropicVorticity{𝒯, 𝒰, 𝒱, 𝒲}
#    Ω::𝒯
#    Ζ::Matrix{𝒯}
#    ζ::Matrix{𝒯}
#    U::Matrix{𝒯}
#    u::Matrix{𝒯}
#    V::Matrix{𝒯}
#    v::Matrix{𝒯}
#    ∂ζu::Matrix{𝒯}
#    ∂ ζ::Matrix{𝒯}
#    ζₛ
#end

end