using DrWatson
@quickactivate "Simple Spectral PDEs"
push!(LOAD_PATH, srcdir())
using SimpleSpectralPDEs

##

using BenchmarkTools
using ProfileView
using PyPlot
using FFTW

pygui(true)

##

f(x) = exp(-cos(x))
f′(x) = sin(x)*f(x)
f′′(x) = (sin(x)^2 + cos(x))*f(x)

##

N = 4
while N <= 512

    x = gridpoints(N)
    y = f.(x)
    
    y′ = zeros(N)
    y′′ = zeros(N)
    
    F = zeros(ComplexF64, N)
    ∂ = zeros(ComplexF64, N)
    P = plan_fft!(F)
    Pᵢ = plan_ifft!(F)

    copyto!(F, y)
    P * F
    copyto!(∂, F)
    fourier_derivative!(y′, ∂, F, Pᵢ, 1)
    L₂′ = sum((y′ .- f′.(x)).^2)/N |> sqrt

    copyto!(F, y)
    P * F
    copyto!(∂, F)
    fourier_derivative!(y′′, ∂, F, Pᵢ, 2)
    L₂′′ = sum((y′′ .- f′′.(x)).^2)/N |> sqrt    
    
    println("$N $L₂′ $L₂′′")

    N *= 2
end

##

N = 128
x = gridpoints(N)
y = sin.(x)
F = zeros(ComplexF64, N)
F .= y

##

@btime begin
    copyto!($F, $y) 
    fft!($F)
end;

##

P = FFTW.plan_fft!(F, flags=FFTW.PATIENT)

@btime begin 
    copyto!($F, $y) 
    $P * $F
end;

##
