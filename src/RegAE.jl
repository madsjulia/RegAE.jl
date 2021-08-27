module RegAE

import BSON
import Distributed
import JLD2
import NNlib
import Optim
import Zygote

include("vae.jl")

mutable struct Autoencoder{T, S, F, M, D}
    encoder::T#maps parameter space (normalized, flattened) to latent space
    decoder::S#maps latent space to (normalized, flattened) paramter space
	mean_data::F#the mean of the training data
	sigma_data::F#the standard deviation of the training data
	mean_latent::M#the mean of the test data in latent space
	cov_latent::D#the covariance matrix of the test data in latent space
end

#train (or load) the autoencoder using data in the variablename variable from data_filename
function Autoencoder(data_filename::String, variablename::String; kwargs...)
	encoder, decoder, mean_data, sigma_data, mean_latent, cov_latent = train(data_filename, variablename; kwargs...)
	encoder = cpu(encoder)
	decoder = cpu(decoder)
    return Autoencoder(encoder, decoder, mean_data, sigma_data, mean_latent, cov_latent)
end

#convert from the parameter space to the latent space
function p2z(ae::Autoencoder, p)
	μ, logσ = ae.encoder(reshape((p .- ae.mean_data) ./ ae.sigma_data, size(p)..., 1))
	z = μ
	return z
end

#convert from the latent space to the parameter space
function z2p(ae::Autoencoder, z)
	p_normalized = ae.decoder(z)
	p = ae.mean_data .+ p_normalized .* ae.sigma_data
	return p
end

#distributed finite difference gradient (if auto-diff is not used)
function gradient(z, objfunc, h)
	zs = map(i->copy(z), 1:length(z) + 1)
	for i = 1:length(zs) - 1
		zs[i][i] += h
	end
	ofs = Distributed.pmap(objfunc, zs; batch_size=ceil(length(z) / Distributed.nworkers()))
	return (ofs[1:end - 1] .- ofs[end]) ./ h
end

#do the inverse analysis using finite difference gradients
function optimize(ae::Autoencoder, objfunc, options; h=1e-4, p0=false)
	objfunc_z = z->sum((z - ae.mean_latent) .* (ae.cov_latent \ (z - ae.mean_latent))) + objfunc(z2p(ae, z))
	if p0 == false
        z0 = ae.mean_latent
	else
		z0 = p2z(ae, p0)
	end
	opt = Optim.optimize(objfunc_z, z->gradient(z, objfunc_z, h), z0, Optim.LBFGS(), options; inplace=false)
	return z2p(ae, opt.minimizer), opt
end

#do the inverse analysis using automatically differentiated gradients
function optimize_zygote(ae::Autoencoder, objfunc, options; p0=false)
	objfunc_z = z->sum((z - ae.mean_latent) .* (ae.cov_latent \ (z - ae.mean_latent))) + objfunc(z2p(ae, z))
	if p0 == false
        z0 = ae.mean_latent
	else
		z0 = p2z(ae, p0)
	end
	opt = Optim.optimize(objfunc_z, z->Zygote.gradient(objfunc_z, z)[1], z0, Optim.LBFGS(), options; inplace=false)
	return z2p(ae, opt.minimizer), opt
end

end
