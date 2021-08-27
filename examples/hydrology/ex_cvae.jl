using Distributed
@everywhere using Flux
using LaTeXStrings
@everywhere import NNlib
import Optim
import PyPlot
@everywhere import RegAE
import StatsBase
import Zygote

@everywhere variablename = "allloghycos"
@everywhere datafilename = "$(results_dir)/trainingdata.jld2"
if !isfile(datafilename)
	if nworkers() == 1
		error("Please run in parallel: julia -p 32")
	end
	numsamples = 10^6
	@time allloghycos = SharedArrays.SharedArray{Float32}(numsamples, ns[2], ns[1]; init=A->samplehyco!(A; setseed=true))
	@time @JLD2.save datafilename allloghycos
end

@everywhere function trainvae(latent_dim)
	CUDA.device!(mod(myid(), length(CUDA.devices())))
	hidden_dim = 5 * latent_dim
	encoder = RegAE.Encoder(Chain(x->reshape(x, 100, 100, 1, size(x, 2)),
								  Conv((4, 4), 1 => 32, relu; stride = 2, pad = 1),
								  Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
								  Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
								  Flux.flatten,
								  Dense(12 * 12 * 32, 256, relu),
								  Dense(256, hidden_dim, relu)),
							Dense(hidden_dim, latent_dim),
							Dense(hidden_dim, latent_dim))
	decoder = Chain(Dense(latent_dim, hidden_dim, relu),
					Dense(hidden_dim, 256, relu),
					Dense(256, 4608, relu),
					x->reshape(x, 12, 12, 32, :),
					ConvTranspose((4, 4), 32 => 32, relu; stride = 2, pad = 1),
					ConvTranspose((4, 4), 32 => 32, relu; stride = 2, pad = 0),
					ConvTranspose((4, 4), 32 => 1; stride = 2, pad = 1),
					Flux.flatten)
	x = RegAE.Autoencoder(datafilename, variablename; model_path="$(results_dir)/vae_nz$(latent_dim).bson", opt=Flux.ADAM(1e-3), epochs=100, seed=1, latent_dim=latent_dim, hidden_dim=hidden_dim, input_dim=10^4, batch_size=100, image_dir="$(results_dir)/images_cvae_$(latent_dim)", encoder=encoder, decoder=decoder, results_dir=results_dir)
	return x
end
latent_dims = [25, 50, 100, 200]
if length(CUDA.devices()) > nworkers()
	error("Now you want to run this part with julia -p $(length(CUDA.devices()))")
end
try
	pmap(trainvae, latent_dims)
catch
	map(trainvae, latent_dims)
end

@everywhere Random.seed!(0)
p_trues = Array(SharedArrays.SharedArray{Float32}(3, ns[2], ns[1]; init=samplehyco!))
casenames = ["nz$i" for i in latent_dims]
for i_p in 1:size(p_trues, 1)
	for i_case in 1:length(casenames)
		if !isfile("$(results_dir)/opt_$(i_p)_$(i_case).jld2")
			casename = casenames[i_case]
			p_true = p_trues[i_p, :, :]
			ae = RegAE.Autoencoder("", ""; model_path="$(results_dir)/vae_$(casename).bson")#this should just load the vae from the bson file
			indices = reshape(collect(1:10^4), 100, 100)
			obsindices = indices[17:17:100, 17:17:100][:]
			p_indices = reshape(collect(1:10^4), 100, 100)
			p_obsindices = p_indices[17:17:100, 17:17:100][:]
			logKs2Ks_neighbors(Ks) = exp.(0.5 * (Ks[map(p->p[1], neighbors)] .+ Ks[map(p->p[2], neighbors)]))
			sources = zeros(size(coords, 2))
			function gethead(p)
				loghycos = p
				if maximum(loghycos) - minimum(loghycos) > 25
					return fill(NaN, length(sources))#this is needed to prevent the solver from blowing up if the line search takes us somewhere crazy
				else
					neighborhycos = logKs2Ks_neighbors(loghycos)
					head = DPFEHM.groundwater_steadystate(neighborhycos, neighbors, areasoverlengths, dirichletnodes, dirichletheads, sources; reltol=1e-12)
					if maximum(head) > lefthead + 1e-4 || minimum(head) < righthead - 1e-4
						error("problem with solution -- head out of range")
					end
					return head
				end
			end
			head_true = reshape(gethead(p_true), size(p_true)...)
			function objfunc(p_flat; show_parts=false)
				p = reshape(p_flat, size(p_true)...)
				head = reshape(gethead(p), size(head_true)...)
				if show_parts
					@show sqrt(sum((head[obsindices] .- head_true[obsindices]) .^ 2) / length(obsindices))
					@show sqrt(sum((p_true[p_obsindices] .- p[p_obsindices]) .^ 2) / length(p_obsindices))
				end
				return 1e4 * sum((head[obsindices] .- head_true[obsindices]) .^ 2) + 3e0 * sum((p_true[p_obsindices] .- p[p_obsindices]) .^ 2)
			end
			@time p_flat, opt = RegAE.optimize_zygote(ae, objfunc, Optim.Options(iterations=200, extended_trace=false, store_trace=true, show_trace=true))
			@JLD2.save "$(results_dir)/opt_$(i_p)_$(i_case).jld2" p_true p_flat opt
			fig, axs = PyPlot.subplots(1, 2)
			axs[1].imshow(reshape(RegAE.z2p(ae, opt.minimizer), size(p_true)...), cmap="jet", extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest")
			p = RegAE.z2p(ae, opt.minimizer)
			@show minimum(p), maximum(p)
			axs[2].imshow(p_trues[i_p,: ,:], cmap="jet", extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest")
			display(fig)
			println()
			fig.savefig("$(results_dir)/result_$(i_case)_$(i_p).pdf")
  			PyPlot.close(fig)
  		end
  	end
end

representations =[latexstring("n_z=$(latent_dim)") for latent_dim in latent_dims]
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
fig, axs = PyPlot.subplots(length(latent_dims) + 1, 3; figsize = (12, 20))
k = 1
for i_p in 1:size(p_trues, 1)
	global k
		@JLD2.load "$(results_dir)/opt_$(i_p)_1.jld2" p_true
		vmin, vmax = extrema(p_trues[i_p, :, :])
		axs[1, i_p].imshow(p_trues[i_p, :, :], vmin=vmin, vmax=vmax, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet")
		axs[1, i_p].set_aspect("equal")
		axs[1, i_p].set_title("Reference Field")
		axs[1, i_p].set_ylabel(L"y")
		axs[1, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
		k += 1
		for i_case in 1:length(casenames)
			@JLD2.load "$(results_dir)/opt_$(i_p)_$(i_case).jld2" p_flat
			p_opt = reshape(p_flat, size(p_trues[1, : ,:]) ...)
			@show casenames[i_case]
			axs[i_case + 1, i_p].imshow(p_opt, vmin=vmin, vmax=vmax, extent=[-sidelength, sidelength, -sidelength, sidelength], origin="lower", interpolation="nearest", cmap="jet")
			axs[i_case + 1, i_p].set_aspect("equal")
			axs[i_case + 1, i_p].set_title(representations[i_case] * ", Relative Error: $(round(sum((p_opt .- p_trues[i_p, :, :]) .^ 2) / sum((p_trues[i_p, :, :] .- StatsBase.mean(p_trues[i_p, :, :])) .^ 2); digits=2))")
			axs[i_case + 1, i_p].set_ylabel(L"y")
			axs[i_case + 1, i_p].set_xlabel(L"x" * "\n($(alphabet[k]))")
			k += 1
	end
end
fig.tight_layout()
display(fig)
fig.savefig("$(results_dir)/megaplot2.pdf")
println()
PyPlot.close(fig)

fig, axs = PyPlot.subplots(1, 3; figsize=(12, 4))
k = 1
for i_p in 1:size(p_trues, 1)
	global k
	for i_case in 1:length(casenames)
		@JLD2.load "$(results_dir)/opt_$(i_p)_$(i_case).jld2" opt
		axs[i_p].semilogy(map(t->t.iteration, opt.trace), map(t->t.value, opt.trace), label=representations[i_case], lw=3, alpha=0.5)
	end
	axs[i_p].legend()
	axs[i_p].set_ylabel("Objective Function")
	axs[i_p].set_xlabel("Iteration\n($(alphabet[k]))")
	k += 1
end
fig.tight_layout()
display(fig)
fig.savefig("$(results_dir)/convergence.pdf")
println()
PyPlot.close(fig)
