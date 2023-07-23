# VLBILikelihoods

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ptiede.github.io/VLBILikelihoods.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptiede.github.io/VLBILikelihoods.jl/dev/)
[![Build Status](https://github.com/ptiede/VLBILikelihoods.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ptiede/VLBILikelihoods.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ptiede/VLBILikelihoods.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ptiede/VLBILikelihoods.jl)


This package has a number of high-performance likelihoods necessary for VLBI modeling. This includes

 - Complex Visibilties
 - Coherency matrices
 - Amplitudes
 - Log Closure Amplitudes (full covariance and diagonal)
 - Closure Phases (full covariance and diagonal)
 
 Currently, most of the distributions will only work on the CPU, but in the future, we expect to adjust these to work on accelerated platforms as well.
