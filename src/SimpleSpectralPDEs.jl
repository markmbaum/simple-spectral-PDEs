module SimpleSpectralPDEs

using FFTW
using UnPack
using OrdinaryDiffEq

export evaluate_terms!, ∂u!

#------------------------------------------------------------------------------
# general

export fourier_derivative!, gridpoints, randominit

function fourier_derivative!(F::Vector{𝒯})::Nothing where {𝒯 <: Complex}
	N = length(F)
	@assert ispow2(N)
	@inbounds for k ∈ 1:N÷2
		F[k] *= im * (k - 1)
	end
	@inbounds for k ∈ N÷2+1:N
		F[k] *= im * (k - 1 - N)
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

function gridpoints(N::Int, 𝒯 = Float64)
	x = LinRange(0, 2π, N + 1)[1:end-1]
	x = collect(𝒯, x)
	return x
end

function randominit(model::𝒯, nmax::Int = 8) where {𝒯}
	@unpack N, Pᵢ = model
	𝒰 = fieldtype(𝒯, 1) |> eltype
	#load random values into a spectral vector
	@assert nmax + 1 ≤ N
	y = zeros(Complex{𝒰}, N)
	for n ∈ 2:nmax+1
		y[n] = randn(Complex{𝒰}) / exp2(√n)
	end
	#take the ifft and return reals
	Pᵢ * y
	return y .|> real
end

#------------------------------------------------------------------------------
# simple advection-diffusion Equation

export AdvectionDiffusion

struct AdvectionDiffusion{𝒯, 𝒰, 𝒱, 𝒲}
	x::Vector{𝒯}
	F::Vector{Complex{𝒯}}
	∂::Vector{Complex{𝒯}} #staging vector for fourier derivatives
	uₓ::Vector{𝒯}
	uₓₓ::Vector{𝒯}
	D::𝒯
	N::Int64
	P::𝒰 #fft plan
	Pᵢ::𝒱 #ifft plan
	𝓋::𝒲 #velocity function 𝓋(x,t)
end

function Base.show(io::IO, model::AdvectionDiffusion{𝒯}) where {𝒯}
	println(io, "$(model.N) point AdvectionDiffusion{$𝒯}")
end

AdvectionDiffusion(; kw...) = AdvectionDiffusion((x, t) -> 1.0; kw...)

AdvectionDiffusion(𝓋::Real; kw...) = AdvectionDiffusion((x, t) -> 𝓋; kw...)

function AdvectionDiffusion(𝓋::Function; D = 0.0, N::Int = 128, 𝒯::Type = Float64) # 𝓋(x,t)
	checksetup(N, 𝒯)
	x = gridpoints(N, 𝒯)
	F = zeros(Complex{𝒯}, N)
	∂ = zeros(Complex{𝒯}, N)
	uₓ = zeros(𝒯, N)
	uₓₓ = zeros(𝒯, N)
	P = plan_fft!(F, flags = FFTW.PATIENT)
	Pᵢ = plan_ifft!(F, flags = FFTW.PATIENT)
	AdvectionDiffusion(x, F, ∂, uₓ, uₓₓ, convert(𝒯, D), N, P, Pᵢ, 𝓋)
end

function evaluate_terms!(model::AdvectionDiffusion{𝒯}, u::AbstractVector{𝒯}) where {𝒯}
	#unpack model arrays
	@unpack F, ∂, uₓ, uₓₓ, N, P, Pᵢ = model
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
	#second derivative
	fourier_derivative!(uₓₓ, ∂, F, Pᵢ, 1)
	return nothing
end

advection_diffusion(x, t, uₓ, uₓₓ, D, 𝓋::ℱ) where {ℱ} = -𝓋(x, t) * uₓ + D * uₓₓ

function ∂u!(∂u, u, model::AdvectionDiffusion, t)::Nothing
	@unpack x, uₓ, uₓₓ, D, 𝓋 = model
	evaluate_terms!(model, u)
	@inbounds for i ∈ eachindex(∂u)
		∂u[i] = advection_diffusion.(x[i], t, uₓ[i], uₓₓ[i], D, 𝓋)
	end
	nothing
end

#------------------------------------------------------------------------------
# Korteweg-De Vries model

export KortewegDeVries

struct KortewegDeVries{𝒯, 𝒰, 𝒱}
	a::𝒯
	F::Vector{Complex{𝒯}}
	∂::Vector{Complex{𝒯}} #staging vector for fourier derivatives
	uₓ::Vector{𝒯}
	uₓₓ::Vector{𝒯}
	uₓₓₓ::Vector{𝒯}
	D::𝒯
	N::Int64
	P::𝒰 #fft plan
	Pᵢ::𝒱 #ifft plan
end

function Base.show(io::IO, model::KortewegDeVries{𝒯}) where {𝒯}
	println(io, "$(model.N) point KortewegDeVries{$𝒯} with a=$(model.a)")
end

function KortewegDeVries(; a = 0.1, N::Int = 128, D::Real = 0.0, 𝒯::Type = Float64)
	checksetup(N, 𝒯)
	F = zeros(Complex{𝒯}, N)
	∂ = zeros(Complex{𝒯}, N)
	uₓ = zeros(𝒯, N)
	uₓₓ = zeros(𝒯, N)
	uₓₓₓ = zeros(𝒯, N)
	P = plan_fft!(F, flags = FFTW.PATIENT)
	Pᵢ = plan_ifft!(F, flags = FFTW.PATIENT)
	KortewegDeVries(convert(𝒯, a), F, ∂, uₓ, uₓₓ, uₓₓₓ, convert(𝒯, D), N, P, Pᵢ)
end

function evaluate_terms!(model::KortewegDeVries{𝒯}, u::AbstractVector{𝒯}) where {𝒯}
	#unpack model arrays
	@unpack F, ∂, uₓ, uₓₓ, uₓₓₓ, D, N, P, Pᵢ = model
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
	if D != zero(𝒯)
		#second derivative
		fourier_derivative!(uₓₓ, ∂, F, Pᵢ, 1)
		#third derivative
		fourier_derivative!(uₓₓₓ, ∂, F, Pᵢ, 1)
	else
		#third derivative
		fourier_derivative!(uₓₓₓ, ∂, F, Pᵢ, 2)
	end
	return nothing
end

korteweg_de_vries(u, uₓ, uₓₓₓ, a, uₓₓ, D) = -u * uₓ - a * a * uₓₓₓ + D * uₓₓ

function ∂u!(∂u, u, model::KortewegDeVries, t)::Nothing
	@unpack uₓ, uₓₓ, uₓₓₓ, a, D = model
	evaluate_terms!(model, u)
	∂u .= korteweg_de_vries.(u, uₓ, uₓₓₓ, a, uₓₓ, D)
	nothing
end

#------------------------------------------------------------------------------
# Kuramoto-Sivashinsky model

export KuramotoSivashinsky

struct KuramotoSivashinsky{𝒯, 𝒰, 𝒱}
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

function KuramotoSivashinsky(; L = 2.5, N::Int = 128, 𝒯::Type = Float64)
	checksetup(N, 𝒯)
	F = zeros(Complex{𝒯}, N)
	∂ = zeros(Complex{𝒯}, N)
	uₓ = zeros(𝒯, N)
	uₓₓ = zeros(𝒯, N)
	uₓₓₓₓ = zeros(𝒯, N)
	P = plan_fft!(F, flags = FFTW.PATIENT)
	Pᵢ = plan_ifft!(F, flags = FFTW.PATIENT)
	KuramotoSivashinsky(convert(𝒯, L), F, ∂, uₓ, uₓₓ, uₓₓₓₓ, N, P, Pᵢ)
end

function evaluate_terms!(model::KuramotoSivashinsky{𝒯}, u::AbstractVector{𝒯}) where {𝒯}
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

kuramoto_sivashinsky(u, uₓ, uₓₓ, uₓₓₓₓ) = -(uₓₓ + uₓₓₓₓ + u * uₓ)

function ∂u!(∂u, u, model::KuramotoSivashinsky, t)::Nothing
	@unpack uₓ, uₓₓ, uₓₓₓₓ = model
	evaluate_terms!(model, u)
	∂u .= kuramoto_sivashinsky.(u, uₓ, uₓₓ, uₓₓₓₓ)
	nothing
end

#------------------------------------------------------------------------------
# integration

export integrate

#convenience/barrier
function integrate(model, u₀, tspan; method::Symbol = :QNDF, tol = 1e-9)
	integrate(model, u₀, tspan, method |> eval, tol)
end

function integrate(model, u₀, tspan, method, tol)
	solve(
		ODEProblem(∂u!, u₀, tspan, model),
		method(autodiff = false),
		reltol = tol,
	)
end

end
