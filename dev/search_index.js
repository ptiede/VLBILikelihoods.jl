var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = VLBILikelihoods","category":"page"},{"location":"#VLBILikelihoods","page":"Home","title":"VLBILikelihoods","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for VLBILikelihoods.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [VLBILikelihoods]","category":"page"},{"location":"#VLBILikelihoods.AmplitudeLikelihood-Tuple{AbstractVector, AbstractVector}","page":"Home","title":"VLBILikelihoods.AmplitudeLikelihood","text":"AmplitudeLikelihood(μ, Σ::Union{AbstractVector, Diagonal})\n\nForms the likelihood for amplitudes from the mean vector μ and the covariance matrix Σ. If Σ is vector or a diagonal matrix then we assume that the argument is the diagonal covariance matrix. If Σ is a full matrix then we assume that a dense covariance was passed\n\nNotes\n\nWe do no processing to the data, i.e. the mean μ is not-debiased anywhere.\n\nWarning\n\nThis likelihood will be significantly biased from the true Rice distribution for data points with SNR = μ/Σ < 2. If this matter significantly for you, we recommend that you consider fitting pure complex visibilities instead.\n\n\n\n\n\n","category":"method"}]
}