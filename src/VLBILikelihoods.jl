module VLBILikelihoods


using Bessels
using DensityInterface
using Distributions
const Dists = Distributions
using PDMats
using LinearAlgebra
using ChainRulesCore
using MappedArrays

using Distributions: log2π

export logdensityof, unnormed_logpdf

# Write your package code here.

abstract type AbstractVLBIDistributions <:  Dists.ContinuousMultivariateDistribution end

lognorm(d::AbstractVLBIDistributions)  = d.lognorm

function Dists.logpdf(d::AbstractVLBIDistributions, x::AbstractVector)
    @assert Dists.insupport(d, x)
    return unnormed_logpdf(d, x) + lognorm(d)
end

include("utility.jl")
include("amplitude.jl")
include("closure_phase.jl")
include("complexvis.jl")


end
