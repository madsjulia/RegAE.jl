import BSON
using CUDA
import FileIO
using Flux
using Flux: @functor
using Flux.Losses: logitbinarycrossentropy, binarycrossentropy
using Flux.Data: DataLoader
import Random
using PyPlot
import Statistics

function loaddata(filename, variablename; batch_size=128)
	data = FileIO.load(filename, variablename)
	numpixels = size(data, 2)
	xtrn = Array{Float32}(undef, numpixels, numpixels, 1, div(8 * size(data, 1), 10))
    xtst = Array{Float32}(undef, numpixels, numpixels, 1, size(data, 1) - div(8 * size(data, 1), 10))
	for i = 1:size(xtrn, 4)
        xtrn[:, :, 1, i] = data[i, :, :]
    end
    mean_data = Statistics.mean(xtrn)
    sigma_data = Statistics.std(xtrn)
    xtrn = (xtrn .- mean_data) ./ sigma_data
    for i = 1:size(xtst, 4)
        xtst[:, :, 1, i] = (data[i + div(8 * size(data, 1), 10), :, :] .- mean_data) ./ sigma_data
    end
    xtrn = reshape(xtrn, 100^2, :)
    xtst = reshape(xtrn, 100^2, :)
    return DataLoader((xtrn,), batchsize=batch_size, shuffle=true), DataLoader((xtst,), batchsize=batch_size, shuffle=true), mean_data, sigma_data
end

struct Encoder
    linear
    μ
    logσ
end
@functor Encoder

Encoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Encoder(
    Dense(input_dim, hidden_dim, tanh),   # linear
    Dense(hidden_dim, latent_dim),        # μ
    Dense(hidden_dim, latent_dim),        # logσ
)

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
)

function reconstruct(encoder, decoder, x, noise)
    μ, logσ = encoder(x)
    z = μ + noise .* exp.(logσ)
    μ, logσ, decoder(z)
end

const F = Float32
function makeloss(encoder, decoder)
    ps = Flux.params(decoder)
    function loss(x, noise)
        mu, logsigma, decoder_z = reconstruct(encoder, decoder, x, noise)
        len = size(x)[end]
        kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logsigma) + mu^2 - 1f0 - 2f0 * logsigma)) / len
        mse = sum((decoder_z .- x) .^ 2) / len
        return kl_q_p + mse
    end
end

function train(filename, variablename; model_path="model.bson", opt=ADAM(1e-4), epochs=2000, seed=1, latent_dim=200, hidden_dim=500, input_dim=10^4, batch_size=100, image_dir="images", encoder=Encoder(input_dim, latent_dim, hidden_dim), decoder=Decoder(input_dim, latent_dim, hidden_dim), results_dir=".")
    if !isfile(model_path)
        seed > 0 && Random.seed!(seed)
        gpu_or_cpu = gpu
        # load train data
        train_loader, test_loader, mean_data, sigma_data = loaddata(filename, variablename; batch_size=batch_size)
        # initialize encoder and decoder
        encoder = encoder |> gpu_or_cpu
        decoder = decoder |> gpu_or_cpu
        # parameters
        ps = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)
        # training
        loss = makeloss(encoder, decoder)
        test_losses = Float64[]
        train_losses = Float64[]
        for epoch = 1:epochs
            train_loss = 0.0
            for (x,) in train_loader
                noise = randn(Float32, latent_dim, batch_size) |> gpu_or_cpu
                x_device = x |> gpu_or_cpu
                train_loss += loss(x_device, noise)
                _, back = Flux.pullback(()->loss(x_device, noise), ps)
                grad = back(1f0)
                Flux.Optimise.update!(opt, ps, grad)
            end
            if mod(epoch, 1) == 0
                #output some info about the training progress, and save the model if it is the best yet
                test_loss = 0.0
                for (x,) in test_loader
                    noise = randn(Float32, latent_dim, batch_size) |> gpu_or_cpu
                    x_device = x |> gpu_or_cpu
                    test_loss += loss(x_device, noise)
                end
                push!(test_losses, test_loss)
                push!(train_losses, train_loss)
                BSON.@save "$(results_dir)/losses.bson" test_losses train_losses
                #plot an example reconstruction and the losses
                fig, axs = PyPlot.subplots(1, 3, figsize=(24, 9))
                x = train_loader.data[1][:, 1]
                axs[1].imshow(reshape(x, 100, 100), vmin=minimum(x), vmax=maximum(x))
                axs[1].set_title("nz=$(latent_dim)")
                rx = cpu(reconstruct(encoder, decoder, train_loader.data[1][:, 1] |> gpu_or_cpu, randn(Float32, latent_dim, 1) |> gpu_or_cpu)[3][:, 1])
                axs[2].imshow(reshape(rx, 100, 100), vmin=minimum(x), vmax=maximum(x))
                axs[2].set_title("Relative RMSE: $(sqrt(sum((x .- rx) .^ 2) / sum(rx .^ 2)))")
                axs[3].plot(test_losses, label="test")
                axs[3].plot(train_losses, label="train")
                axs[3].set(xlabel="epoch", ylabel="loss")
                axs[3].legend()
                if !isdir(image_dir)
                    mkdir(image_dir)
                end
                fig.savefig("$(image_dir)/reconstruction_$(lpad(epoch, 4, '0')).png")
                PyPlot.close(fig)
                if test_loss == minimum(test_losses)
                    save_model(model_path, encoder, decoder, mean_data, sigma_data, batch_size, test_loader, latent_dim, gpu_or_cpu)
                end
            end
        end
    end
    BSON.@load model_path encoder decoder mean_data sigma_data mean_latent sigma_latent
    return encoder, decoder, mean_data, sigma_data, mean_latent, sigma_latent
end

function save_model(model_path, encoder, decoder, mean_data, sigma_data, batch_size, test_loader, latent_dim, gpu_or_cpu)
    test_zs = zeros(latent_dim, length(test_loader) * batch_size)
    i = 1
    for (x,) in test_loader
        z = cpu(encoder(x |> gpu_or_cpu)[1])
        test_zs[:, i:i + batch_size - 1] = z
        i += batch_size
    end
    mean_latent = vec(Statistics.mean(test_zs; dims=2))
    sigma_latent = Statistics.cov(test_zs; dims=2)
    let encoder = cpu(encoder), decoder = cpu(decoder)
        BSON.@save model_path encoder decoder mean_data sigma_data mean_latent sigma_latent
    end
    return encoder, decoder, mean_data, sigma_data, mean_latent, sigma_latent
end
