using Pkg
Pkg.activate(".")

using Flux, Flux.Losses
using Statistics

using Transformers, Transformers.Layers
using CSV, BSON # , JSON
using Functors
using BenchmarkTools
using Random
using ProgressMeter

using Sound
using WAV
using DSP
using FFTW
using LinearAlgebra

using PyPlot; pygui(true)
using CUDA; enable_gpu(CUDA.functional())

const TRAIN= true

d_in=129                         # Input data Dimention
d_out=d_in                       # Output data Dimention

n_layers = 4                     # Number of transformer layers
n_heads = 4                      # Number of parallel transformer heads (8)
d_k = 64                         # Dimension of the attention operation d_v=d_q=d_k in Transformer.jl. (32)
d= n_heads * d_k                 # Hiddeen dim, i.e. embedding, output, internal representation
d_ffn = 4d                       # Dimention of feed forward layer

encoding_transformer = Transformer(TransformerBlock, n_layers, n_heads, d, d_k, d_ffn) |> todevice  #; dropout=10

position_embedding = SinCosPositionEmbed(d)
projection_layer=Flux.Dense(d_in,d,identity)  |> todevice
antiprojection_layer=Flux.Dense(d, d_out,identity)  |> todevice

function withmask(input)
    masks = Masks.LengthMask(input.mask)
    return (y=input.y, x=(hidden_state=input.x, attention_mask=masks))
end

struct Model
    position_embedding
    projection
    withmask 
    transformer
    antiprojection
end
 
model=Model(position_embedding, projection_layer, withmask, encoding_transformer, antiprojection_layer)
@functor Model (projection, transformer, antiprojection)

function (model::Model)(input)
    position= model.position_embedding(input.hidden_state)
    projection = model.projection(input.hidden_state)
    transformed=model.transformer( (hidden_state=projection .+ position, attention_mask=input.attention_mask) )
    result=(hidden_state=model.antiprojection(transformed.hidden_state),attention_mask=transformed.attention_mask) 
    return result
end

function loss(model, input)
    y=input.y
    x=input.x
    ŷ=model(x)
    mask=ŷ.attention_mask
    return Flux.mse(mask .*y, mask .* ŷ.hidden_state)
end

Random.seed!(7) 
include("./read_audio.jl")
include("./audio_processing.jl")

n_samples=16384
stepsize=12288
fs=8192

noises=read_noises("./data/environmental_sound_classification_50/archive/esc50.csv", "frog", "./data/environmental_sound_classification_50/archive/audio/audio/16000/", fs) |> 
    shuffle_n_split |>
    splits -> overlapping_splits(splits, n_samples, stepsize)

signals=read_signals("./data/fluent_speech_commands_dataset/data/test_data.csv", "./data/fluent_speech_commands_dataset/", "7B4XmNppyrCK977p", fs, n_samples) |>
    shuffle_n_split

win=hanning(256+1)[2:end]
m=length(win)

f_low=300
f_high=fs>>1
M=filter_bank(d_in, m, fs, f_low, f_high)

batchsize=32

train=dataloader(signals.train,noises.train, win, M; batchsize=batchsize, shuffle=true)
test=(oos=dataloader(signals.test,noises.test, win, M; batchsize=batchsize, shuffle=true),
      is= dataloader(signals.train[1:length(signals.test)], noises.train[1:length(noises.test)], win, M; batchsize=batchsize, shuffle=true))


dataloaderloss(loader, model=model)=mean([loss(model,withmask(batch)) for batch in loader])
losses=[dataloaderloss(test.is, identity) dataloaderloss(test.oos, identity)]
println("0: $losses")

optimizerstate = Flux.setup(Adam(5e-4), model)
epochs=1:8

function train!()
    @info "start training"
    global losses
    for epoch in epochs
        @showprogress for (i,sample) in enumerate(train)
            input=withmask(sample)
            ϵ,∇=Flux.withgradient(model, input) do m, s
                loss(m, s)
            end
            Flux.update!(optimizerstate, model, ∇[1])             
        end
        ls=[dataloaderloss(test.is) dataloaderloss(test.oos)]
        println("$epoch: $ls")
        losses=vcat(losses,ls)
    end
end

if TRAIN
    @time train!()
    model=Model(model.position_embedding, model.projection, withmask, model.transformer, model.antiprojection) |> cpu
    signals_n_noises=(signals,noises,losses)
    BSON.@save "model.bson" model
    BSON.@save "signals_n_noises.bson" signals_n_noises
    model=model |> todevice
else
    BSON.@load "signals_n_noises.bson" signals_n_noises
    (signals,noises,losses)=signals_n_noises
    BSON.@load "model.bson" model
    model=model |> todevice
end

clf()
plot(losses[2:end,1]./losses[1,1])
plot(losses[2:end,2]./losses[1,2])

title("Training loss")
legend(["is","oos"])
ylabel("loss")
xlabel("epoch")
grid()

savefig("training.png")
