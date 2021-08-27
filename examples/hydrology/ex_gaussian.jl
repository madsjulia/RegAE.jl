@everywhere begin
	import CUDA
	import DPFEHM
	import Flux
	import GaussianRandomFields
	import GeostatInversion
	import JLD2
	import Random
	import RegAE
	import SharedArrays
	import Statistics

	Random.seed!(myid())
	results_dir = "gaussian_results"
	if !isdir(results_dir) && myid() == 1
		mkdir(results_dir)
	end

	sidelength = 50.0#m
	thickness = 10.0#m
	mins = [-sidelength, -sidelength, 0]
	maxs = [sidelength, sidelength, thickness]
	ns = [100, 100, 1]
	meanloghyco1 = log(1e-5)#m/s
	lefthead = 1.0#m
	righthead = 0.0#m
 	coords, neighbors, areasoverlengths, _ = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)

	lambda1 = 50.0
	sigma1 = 1.0
	cov1 = GaussianRandomFields.CovarianceFunction(2, GaussianRandomFields.Matern(lambda1, 1; σ=sigma1))
	num_eigenvectors = 200
	xs = range(mins[1]; stop=maxs[1], length=ns[1])
	ys = range(mins[2]; stop=maxs[2], length=ns[2])
	zs = range(mins[3]; stop=maxs[3])
	grf1 = GaussianRandomFields.GaussianRandomField(cov1, GaussianRandomFields.KarhunenLoeve(num_eigenvectors), xs, ys)
	function samplehyco!(fields::SharedArrays.SharedArray; setseed=false)
		if nworkers() == 1 || size(fields, 1) == 3#if it is small or there is only one processor break up the chunk more simply
			if myid() <= 2
				mychunk = 1:size(fields, 1)
			else
				mychunk = 1:0
			end
		else
			mychunk = 1 + div((myid() - 2) * size(fields, 1), nworkers()):div((myid() - 1) * size(fields, 1), nworkers())
		end
		for i in mychunk
			if setseed
				Random.seed!(i)
			end
			fields[i, :, :] = meanloghyco1 .+ GaussianRandomFields.sample(grf1)'
		end
		return nothing
	end

	boundaryhead(x, y) = (lefthead-righthead) * (x - maxs[1]) / (mins[1] - maxs[1])
	dirichletnodes = Int[]
	dirichletheads = zeros(size(coords, 2))
	for i = 1:size(coords, 2)
		if coords[1, i] == mins[1] || coords[1, i] == maxs[1]
			push!(dirichletnodes, i)
			dirichletheads[i] = boundaryhead(coords[1:2, i]...)
		end
	end
end

include("ex.jl")
