using Distributed
using LaTeXStrings
import Knet
import Optim
import PyPlot
import StatsBase

@everywhere begin
	import FiniteVolume
	import GeostatInversion
	import JLD2
	import Random
	import RegAE
	import SharedArrays
	import Statistics

	Random.seed!(myid())

	sidelength = 50.0#m
	thickness = 10.0#m
	mins = [-sidelength, -sidelength, 0]
	maxs = [sidelength, sidelength, thickness]
	ns = [100, 100, 2]
	meanloghyco1 = log(1e-5)#m/s
	meanloghyco2 = log(1e-8)#m/s
	lefthead = 1.0#m
	righthead = 0.0#m
	coords, neighbors, areasoverlengths, volumes = FiniteVolume.regulargrid(mins, maxs, ns)

	xs = range(mins[1]; stop=maxs[1], length=ns[1])
	ys = range(mins[2]; stop=maxs[2], length=ns[2])
	zs = range(mins[3]; stop=maxs[3], length=ns[3])
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
			field1 = GeostatInversion.FFTRF.powerlaw_structuredgrid([ns[2], ns[1]], meanloghyco1, 0.5, -3.0)#fairly smooth field
			field2 = GeostatInversion.FFTRF.powerlaw_structuredgrid([ns[2], ns[1]], meanloghyco2, 0.5, -2.5)#fairly rough field
			fieldsplit = GeostatInversion.FFTRF.powerlaw_structuredgrid([ns[2], ns[1]], meanloghyco2, 1, -4.5)#very smooth field
			p = Random.rand()
			splitnumber = Statistics.quantile(fieldsplit[:], 0.25 + 0.5 * p)
			for j1 = 1:size(fields, 2)
				for j2 = 1:size(fields, 3)
					if fieldsplit[j1, j2] > splitnumber
						fields[i, j1, j2] = field2[j1, j2]
					else
						fields[i, j1, j2] = field1[j1, j2]
					end
				end
			end
		end
		return nothing
	end
	dirichletnodes = Int[]
	dirichletheads = Float64[]
	for i = 1:size(coords, 2)
		if coords[1, i] == -sidelength
			push!(dirichletnodes, i)
			push!(dirichletheads, lefthead)
		elseif coords[1, i] == sidelength
			push!(dirichletnodes, i)
			push!(dirichletheads, righthead)
		end
	end
end

datafilename = "trainingdata.jld2"
if !isfile(datafilename)
	numsamples = 10^5
	@time allloghycos = SharedArrays.SharedArray{Float32}(numsamples, ns[2], ns[1]; init=A->samplehyco!(A; setseed=true))
	@time @JLD2.save datafilename allloghycos
end
@JLD2.load datafilename allloghycos

ae = RegAE.Autoencoder(datafilename, "vae_nz100.jld2", "--infotime 1 --seed 1 --epochs 10 --nz 100 --nh 500"; varname="allloghycos")
ae = RegAE.Autoencoder(datafilename, "vae_nz200.jld2", "--infotime 1 --seed 1 --epochs 10 --nz 200 --nh 1000"; varname="allloghycos")
ae = RegAE.Autoencoder(datafilename, "vae_nz400.jld2", "--infotime 1 --seed 1 --epochs 10 --nz 400 --nh 2000"; varname="allloghycos")

@everywhere Random.seed!(0)
p_trues = Array(SharedArrays.SharedArray{Float32}(3, ns[2], ns[1]; init=samplehyco!))
casenames = ["nz100", "nz200", "nz400"]
for i_p in 1:size(p_trues, 1)
	for i_case in 1:length(casenames)
		if !isfile("opt_$(i_p)_$(i_case).jld2")
			casename = casenames[i_case]
			p_true = p_trues[i_p, :, :]
			ae = RegAE.Autoencoder("vae_$(casename).jld2")
			indices = reshape(collect(1:2*10^4), 2, 100, 100)
			obsindices = indices[1, 17:17:100, 17:17:100][:]
			p_indices = reshape(collect(1:10^4), 100, 100)
			p_obsindices = p_indices[17:17:100, 17:17:100][:]
			function gethead(p)
				p3d = Array{Float64, 3}(undef, ns[3], ns[2], ns[1])
				p3d[1, :, :] = p
				p3d[2, :, :] = p
				loghycos = reshape(p3d, ns[3], ns[2], ns[1])
				neighborhycos = FiniteVolume.nodehycos2neighborhycos(neighbors, loghycos, true)
				sources = zeros(size(coords, 2))
				head, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, neighborhycos, sources, dirichletnodes, dirichletheads)
				if !ch.isconverged
					error("didn't converge")
				end
				if maximum(head) > lefthead || minimum(head) < righthead
					error("problem with solution -- head out of range")
				end
				return head
			end
			head_true = reshape(gethead(p_true), 2, size(p_true)...)
			function objfunc(p_flat)
				p = reshape(p_flat, size(p_true)...)
				head = reshape(gethead(p), size(head_true)...)
				return 1e4 * sum((head[obsindices] .- head_true[obsindices]) .^ 2) + 3e0 * sum((p_true[p_obsindices] .- p[p_obsindices]) .^ 2)
			end
			@time p_flat, opt = RegAE.optimize(ae, objfunc, Optim.Options(iterations=25, extended_trace=false, store_trace=true, show_trace=false))
			@JLD2.save "opt_$(i_p)_$(i_case).jld2" p_true p_flat opt
			fig, axs = PyPlot.subplots(1, 2)
			axs[1].imshow(reshape(RegAE.z2p(ae, opt.minimizer), size(p_true)...), cmap="jet", vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest")
			axs[2].imshow(p_trues[i_p, :, :], cmap="jet", vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest")
			#display(fig)
			#println()
			fig.savefig("ex3_result_$(i_case)_$(i_p).pdf")
			PyPlot.close(fig)
		end
	end
end


representations = [L"n_z=100", L"n_z=200", L"n_z=400"]
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
fig, axs = PyPlot.subplots(5, 3; figsize=(12, 20))
k = 1
for i_p in 1:size(p_trues, 1)
	global k
	@JLD2.load "opt_$(i_p)_1.jld2" p_true
	axs[1, i_p].imshow(p_trues[i_p, :, :], vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet")
	axs[1, i_p].set_aspect("equal")
	axs[1, i_p].set_title("Reference Field")
	axs[1, i_p].set_ylabel(L"y")
	axs[1, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
	k += 1
	for i_case in 1:length(casenames)
		@JLD2.load "opt_$(i_p)_$(i_case).jld2" p_flat
		p_opt = reshape(p_flat, size(p_trues[1, :, :])...)
		axs[i_case + 1, i_p].imshow(p_opt, vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet")
		axs[i_case + 1, i_p].set_aspect("equal")
		axs[i_case + 1, i_p].set_title(representations[i_case] * ", Relative Error: $(round(sum((p_opt .- p_trues[i_p, :, :]) .^ 2) / sum((p_trues[i_p, :, :] .- StatsBase.mean(p_trues[i_p, :, :])) .^ 2); digits=2))")
		axs[i_case + 1, i_p].set_ylabel(L"y")
		axs[i_case + 1, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
		k += 1
		@JLD2.load "opt_$(i_p)_$(i_case).jld2" opt
		axs[5, i_p].semilogy(map(t->t.iteration, opt.trace), map(t->t.value, opt.trace), label=representations[i_case], lw=3, alpha=0.5)
	end
	axs[5, i_p].legend()
	axs[5, i_p].set_ylabel("Objective Function")
	axs[5, i_p].set_xlabel("Iteration\n($(alphabet[k]))")
	k += 1
end
fig.tight_layout()
display(fig)
fig.savefig("megaplot.pdf")
println()
PyPlot.close(fig)


fig, axs = PyPlot.subplots(2, 3; figsize=(12, 8))
k = 1
for i_p in 1:size(p_trues, 1)
	global k
	@JLD2.load "opt_$(i_p)_1.jld2" p_true
	axs[1, i_p].imshow(p_trues[i_p, :, :], vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet")
	axs[1, i_p].set_aspect("equal")
	axs[1, i_p].set_title("Reference Field")
	axs[1, i_p].set_ylabel(L"y")
	axs[1, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
	k += 1
	for i_case in 1:length(casenames)
		if i_case == 2
			@JLD2.load "opt_$(i_p)_$(i_case).jld2" p_flat
			p_opt = reshape(p_flat, size(p_trues[1, :, :])...)
			axs[i_case, i_p].imshow(p_opt, vmin=ae.lowend, vmax=ae.highend, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet")
			axs[i_case, i_p].set_aspect("equal")
			axs[i_case, i_p].set_title(representations[i_case] * ", Relative Error: $(round(sum((p_opt .- p_trues[i_p, :, :]) .^ 2) / sum((p_trues[i_p, :, :] .- StatsBase.mean(p_trues[i_p, :, :])) .^ 2); digits=2))")
			axs[i_case, i_p].set_ylabel(L"y")
			axs[i_case, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
			k += 1
		end
	end
end
fig.tight_layout()
display(fig)
fig.savefig("comparison.pdf")
println()
PyPlot.close(fig)


fig, axs = PyPlot.subplots(1, 3; figsize=(12, 4))
k = 1
for i_p in 1:size(p_trues, 1)
	global k
	for i_case in 1:length(casenames)
		@JLD2.load "opt_$(i_p)_$(i_case).jld2" opt
		axs[i_p].semilogy(map(t->t.iteration, opt.trace), map(t->t.value, opt.trace), label=representations[i_case], lw=3, alpha=0.5)
	end
	axs[i_p].legend()
	axs[i_p].set_ylabel("Objective Function")
	axs[i_p].set_xlabel("Iteration\n($(alphabet[k]))")
	k += 1
end
fig.tight_layout()
display(fig)
fig.savefig("convergence.pdf")
println()
PyPlot.close(fig)
