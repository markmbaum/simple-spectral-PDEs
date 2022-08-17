
# Simple Spectral PDEs

This small repository implements pseudospectral solvers for a few one-dimensional partial differential equations (PDEs) in periodic domains. In each case, the solver uses in-place fast Fourier transforms and time-steppers from [OrdinaryDiffEq.jl](https://diffeq.sciml.ai/stable/solvers/ode_solve/) for fast, high-accuracy, and stable solutions. The density of the grid can be any power of two. The three PDEs equations implemented are, in order of solutions with increasing complexity:

#### Advection-Diffusion

$$ u_t = -v(x,t)u_x + D u_{xx} $$

where $v(x,t)$ is a velocity function that can be any periodic function of space and time and $D$ is a diffusion coefficient. The example below shows a solution from a random initial condition on a 512 point grid with $v(x,t)=1.25 + \cos(x + 2 \pi t/25)$ and $D=10^{-4}$. The diffusion acts mostly in the migrating low-velocity zone where the ripples are compressed and $u_{xx}$ is highest.

![advection_diffusion](mov/advection_diffusion.gif)

#### Korteweg–De Vries 

$$ u_t = -u u_x - a^2u_{xxx} $$

This is a simplified shallow water wave model with a lot of nonlinearity and a third-order derivative.

![korteweg_de_vries](mov/korteweg_de_vries.gif)

#### Kuramoto-Sivashinsky

$$ u_t = -u_{xx} - u_{xxxx} - u u_{x} $$

A fourth-order PDE famous for its chaotic behavior. The model produces different behavior for different domain length, which causes the terms above to have different scaling.

![kuramoto_sivashinsky](mov/kuramoto_sivashinsky.gif)

--------

More info about pseudospectral techniques generally can be found in the books:
* *Chebyshev and Fourier Spectral Methods* (Boyd)
* *Spectral methods: fundamentals in single domains* (Canuto)
* *A Practical Guide to Pseudospectral Methods* (Fornberg)

--------

This repo uses the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> Simple Spectral PDEs

It is authored by Mark Baum <markmbaum@protonmail.com>.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and everything should work out of the box, including correctly finding local paths.
