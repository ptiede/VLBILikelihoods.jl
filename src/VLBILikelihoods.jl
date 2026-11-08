module VLBILikelihoods

using DocStringExtensions
using Bessels
using EnzymeCore
using EnzymeCore: EnzymeRules, Annotation
using DensityInterface
using Distributions
const Dists = Distributions
using LinearAlgebra
using Random
using ReactantCore: @trace
using SparseArrays
using Serialization
using SuiteSparse
using StructArrays
using StaticArraysCore
const SA = StaticArraysCore

using Distributions: log2π

export logdensityof, unnormed_logpdf, CholeskyFactor

# Write your package code here.

"""
    $(TYPEDEF)

Abstract type that detail that the distribution is the likelihood for some VLBI dataproduct.
One key difference between an `AbstractVLBIDistribution` and a regular `Distributions.jl` one
is that by default we expect on the fields of the `struct` defining the distribution to be
`lognorm` which gives the log-normalization constant for the distribution.

This prevents the model from constantly having to compute the log-normalization constant
everytime the density is evaluated, which is often the most expensive part of the computation.

To implement a `AbstractVLBIDistributions` a user then just needs to define
  - `unormed_logdensity(d::AbstractVLBIDistribution, x)`
which takes in the new distribution type `d` and the point you wish to evaluate the density at.
Internally `VLBILikelihood` adds in the normalization constant.

If the user wishes to change the default lognorm behavior they can also overload the `lognorm`
function to opt-out of storing the normalization constant in the `struct`. For example

struct MyNormal <: AbstractVLBIDistributions end

unnormed_logdensity(d::MyNormal, x) = -x^2/2
lognorm(d::MyNormal) = -log(2π)/2

"""
abstract type AbstractVLBIDistributions <:  Dists.ContinuousMultivariateDistribution end

"""
    lognorm(d::AbstractVLBIDistributions)

Compute the log-normalizatoin constant of the distribution `d`.
"""
lognorm(d::AbstractVLBIDistributions)  = d.lognorm

# Whether `x` is in the support of `d`; every value is, except where a likelihood says otherwise.
_insupport(d::AbstractVLBIDistributions, x) = true

function _logdensity(d::AbstractVLBIDistributions, x)
    @trace if _insupport(d, x)
        l = unnormed_logpdf(d, x) + lognorm(d)
    else
        l = oftype(lognorm(d), -Inf)
    end
    return l
end
DensityInterface.logdensityof(d::AbstractVLBIDistributions, x::AbstractArray) = _logdensity(d, x)
# A matrix is one draw of a likelihood over two dims, not a set of draws.
DensityInterface.logdensityof(d::AbstractVLBIDistributions, x::AbstractMatrix) = _logdensity(d, x)
Dists.logpdf(d::AbstractVLBIDistributions, x::AbstractVector) = _logdensity(d, x)

# One draw has the shape of the mean, whatever its dims.
Dists.rand(rng::Random.AbstractRNG, d::AbstractVLBIDistributions) = Dists._rand!(rng, d, similar(d.μ))

include("cholesky.jl")
include("utility.jl")
include("amplitude.jl")
include("closure_phase.jl")
include("complexvis.jl")
include("coherency.jl")
include("rules.jl")


end
