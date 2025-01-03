using Pkg
Pkg.activate(".")

using Flux, Flux.Losses
using Statistics

using Transformers, Transformers.Layers
using CSV, BSON # , JSON
using Functors
using BenchmarkTools
using Random

using CUDA; enable_gpu(CUDA.functional())

#############################################
#RUN TRAINING DATA
#############################################

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

sample=first(train)
input=withmask(sample)
model(input.x)

function loss(model, input)
    y=input.y
    x=input.x
    ŷ=model(x)
    mask=ŷ.attention_mask
    return Flux.mse(mask .*y, mask .* ŷ.hidden_state)
end

sample=first(test)
loss(model,withmask(sample))

#4.83

optimizerstate = Flux.setup(Adam(1e-4), model)

function train!()
    @info "start training"
    for epoch in 1:3
        for (i,sample) in enumerate(train)
            input=withmask(sample)
            ϵ,∇=Flux.withgradient(model, input) do m, s
                loss(m, s)
            end
            if i % 16 == 0 
                t=loss(model,withmask(first(test)))
                println("p= $(i/length(train)/3+(epoch-1)/3), ϵ=$ϵ, t = $t")
            end
            Flux.update!(optimizerstate, model, ∇[1])             
        end
    end
end
 
@time train!()
ϵ=loss(model,input)

BSON.@load "signals_n_noises.bson" signals_n_noises
(signals,noises)=signals_n_noises
BSON.@load "mymodel.bson" model
model=model |> todevice

model=Model(model.position_embedding, model.projection, withmask, model.transformer, model.antiprojection) |> cpu
signals_n_noises=(signals,noises)
using BSON: @save
@save "mymodel.bson" model
@save "signals_n_noises.bson" signals_n_noises