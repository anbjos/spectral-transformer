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

include("./read_audio.jl")
include("./audio_processing.jl")

const TRAIN= false

d_in=26                          # Input data Dimention
d_out=d_in                       # Output data Dimention

n_layers = 2                     # Number of transformer layers
n_heads = 8                      # Number of parallel transformer heads
d_k = 32                         # Dimension of the attention operation d_v=d_q=d_k in Transformer.jl.
d= n_heads * d_k                 # Hiddeen dim, i.e. embedding, output, internal representation
d_ffn = 4d                       # Dimention of feed forward layer

encoding_transformer = Transformer(TransformerBlock, n_layers, n_heads, d, d_k, d_ffn) |> todevice

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

#sample=first(train)
#input=withmask(sample)
#model(input.x)

function loss(model, input)
    y=input.y
    x=input.x
    ŷ=model(x)
    mask=ŷ.attention_mask
    return Flux.mse(mask .*y, mask .* ŷ.hidden_state)
end


#sample=first(test)
#loss(model,withmask(sample))

#4.83

dataloaderloss(loader, model=model)=mean([loss(model,withmask(batch)) for batch in loader])
losses=Matrix{Float32}(undef,0,2)
optimizerstate = Flux.setup(Adam(1e-4), model)
epochs=1:4

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
 
@time train!()
ϵ=loss(model,input)


r=dataloaderloss(test.is, identity)
plot(losses[:,1]/r)
r=dataloaderloss(test.oos, identity)
plot(losses[:,1]/r)

title("baseline\nt=2500s")


legend(["is","oos"])
grid()

savefig("baseline.pdf")

test


xxxxx=loadfig("x.eps")

dataloaderloss(test.is)
dataloaderloss(test.is, reference)
dataloaderloss(test.is, model)
dataloaderloss(test.is, identity)

reference=model

model=reference


dataloaderloss(test.oos)

BSON.@load "signals_n_noises.bson" signals_n_noises
(signals,noises)=signals_n_noises
BSON.@load "model.bson" model
model=model |> todevice

model


model=Model(model.position_embedding, model.projection, withmask, model.transformer, model.antiprojection) |> cpu
signals_n_noises=(signals,noises)
using BSON: @save
@save "model.bson" model
@save "signals_n_noises.bson" signals_n_noises