module VLBILikelihoods

using ComradeBase
using DocStringExtensions
using Bessels
using DensityInterface
using Distributions
const Dists = Distributions
using LinearAlgebra
using ChainRulesCore
using Random
using SparseArrays
using Serialization
using SuiteSparse
using StructArrays
using StaticArraysCore
const SA = StaticArraysCore

using Distributions: log2π

export logdensityof, unnormed_logpdf

# Write your package code here.

"""
    $(TYPEDEF)

Abstract type that detail that the distribution is the likelihood for some VLBI dataproduct.
One key difference between an `AbstractVLBIDistribution` and a regular `Distributions.jl` one
is that by default we expect on the fields of the `struct` defining the distribution to be
`lognorm` which gives the log-normalization constant for the distribution.

This prevents the model from constantly having to compute the log-normalization constant
everytime the density is evaluated, which is often the most expensive part of the computation.

To implement a `AbstractVLBIDistributions` a user then just needs to implement
  - `unormed_logdensity(d::AbstractVLBIDistribution, x)`
which takes in the new distribution type `d` and the point you wish to evaluate the density at.
Internally `VLBILikelihood` add in the normalization constant.

Additionally, if the user wishes to change this behavior they can also overload the `lognorm`
function to opt-out of storing the normalization constant in the `struct`.
"""
abstract type AbstractVLBIDistributions <:  Dists.ContinuousMultivariateDistribution end

"""
    lognorm(d::AbstractVLBIDistributions)

Compute the log-normalizatoin constant of the distribution `d`.
"""
lognorm(d::AbstractVLBIDistributions)  = d.lognorm

# For AbstractVLBI distributions we do a quick adjustment given how we
# have structured the problem.
function Dists.logpdf(d::AbstractVLBIDistributions, x::AbstractVector)
    !Dists.insupport(d, x) && return -Inf
    return unnormed_logpdf(d, x) + lognorm(d)
end

include("cholesky.jl")
include("utility.jl")
include("amplitude.jl")
include("closure_phase.jl")
include("complexvis.jl")
include("coherency.jl")
include("rules.jl")


end
