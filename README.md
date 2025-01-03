# Spectral Transformer

The **Spectral Transformer** repository demonstrates the versatility of transformer-based AI models in signal processing tasks, with a particular focus on **audio applications** like noise suppression. Unlike traditional transformer inputs consisting of tokenized text, this project adapts modern transformer architectures to work directly with **audio signals** represented as mel-spectrograms. The transformers are customized to accommodate this unique input format, enabling highly effective signal analysis and modification.

A key highlight is the integration of neural networks and digital signal processing (DSP) techniques. The results showcase the synergy of these approaches, achieving precise noise suppression while maintaining the **original signal's phase information**. Notably, this approach avoids the use of vocoders for audio reconstruction, ensuring higher fidelity in the output.

Even with a relatively simple transformer network, excellent performance can be achieved in real-world scenarios, as illustrated by the following example. Using the challenging task of separating human speech from background noise (e.g., frog sounds), the network demonstrates its robustness. Specifically:

1. **Signal with Noise** (Figure 1): This represents the input noisy speech signal, containing both human speech and frog sounds.

![Signal with noise](figures/Figure_1.png)

2. **Signal without Noise** (Figure 2): This is the clean speech signal, used as a reference for comparison.

![Signal without noise](figures/Figure_2.png)

3. **Cleaned Signal with Noise** (Figure 3): This is the output of the network, showing its ability to effectively suppress noise and isolate the speech signal, even on out-of-sample data.

![Cleaned signal with noise](figures/Figure_3.png)


## Attention Mechanism
Understanding the [attention mechanism](https://arxiv.org/html/1706.03762v7) is crucial for adapting deep learning architectures to various input formats. While most explanations utilize a sequence × embedding input format, this discussion emphasizes the embedding × sequence format as implemented in Julia. Additionally, this documentation aims to provide a comprehensive overview of the concepts that may initially seem challenging, breaking them down in a clear and accessible manner. By doing so, it not only facilitates modifications to input/output processing but also ensures the provided code is both intuitive and adaptable.

The attention mechanism is a foundational concept in modern deep learning architectures, such as the Transformer. It enables models to focus on specific elements of an input sequence when generating an output, capturing the relationships between elements of the sequence. This section provides a step-by-step explanation of how the attention mechanism operates for **a single attention head**, starting with input representation as matrices, followed by transformations into queries, keys, and values, and concluding with the computation of attention scores and outputs. Each step is described in terms of the matrix operations that underlie the mechanism.

---

### 1. **Input Representation as a Matrix**

The input sequence is represented as a matrix:

Let the sequence have $n$ elements, each represented by a vector of dimension $d$. The input sequence is then represented as:

$$
X \in \mathbb{R}^{d \times n}
$$

where each column corresponds to the representation of an element in the sequence.

---

### 2. **Linear Transformations to Obtain Query, Key, and Value**

Each element of the input sequence is transformed into three separate representations: **query ($Q$)**, **key ($K$)**, and **value ($V$)**. These are computed using learnable weight matrices. Since the input $X$ is $d \times n$, the transformation involves matrix multiplication:

$$
Q = W_Q X \quad \text{where} \quad W_Q \in \mathbb{R}^{d_k \times d}
$$

$$
K = W_K X \quad \text{where} \quad W_K \in \mathbb{R}^{d_k \times d}
$$

$$
V = W_V X \quad \text{where} \quad W_V \in \mathbb{R}^{d_v \times d}
$$

Here:
- $d_k$ is the dimension of the query and key vectors:  

- $d_v$ is the dimension of the value vectors:  

These weight matrices $W_Q$, $W_K$, and $W_V$ are learned during training, with each attention head independently learning its own set of weight matrices.

After the transformations:
- $Q \in \mathbb{R}^{d_k \times n}$,
- $K \in \mathbb{R}^{d_k \times n}$,
- $V \in \mathbb{R}^{d_v \times n}$.

---

### 3. **Attention Score Calculation**

The attention mechanism determines the relevance of each element in the sequence to every other element. This is done by computing a **similarity** score between queries and keys using a [dot product](https://en.wikipedia.org/wiki/Dot_product):

$$
\text{Attention Score} = K^T Q
$$

Where:
- $Q \in \mathbb{R}^{d_k \times n}$ (queries from all elements),
- $K^T \in \mathbb{R}^{n \times d_k}$ (keys transposed).

The result is:

$$
S \in \mathbb{R}^{n \times n}
$$

where $S_{ij}$ represents the relevance of the $j$-th element to the $i$-th element.

To stabilize training and avoid overly large values, these scores are scaled by $\sqrt{d_k}$:

$$
\text{Scaled Attention Score} = \frac{K^T Q}{\sqrt{d_k}}
$$

---

### 4. **Softmax to Compute Attention Weights**

The [softmax](https://en.wikipedia.org/wiki/Softmax_function) function is applied row-wise to the scaled attention scores to produce attention weights, ensuring that each row forms a probability distribution that sums to 1:

$$
A = \text{Softmax}\left(\frac{K^T Q}{\sqrt{d_k}}\right)
$$

For the $i$-th query, the attention weight for the $j$-th key is:

$$
A_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^n \exp(S_{ik})}
$$

This results in the attention weights matrix $A \in \mathbb{R}^{n \times n}$, where each row represents how much each key contributes to the respective query.

---

### 5. **Weighted Sum Using Values**

The attention weights are used to compute a weighted sum of the value vectors:

$$
\text{Output} = V A
$$

Where:
- $V \in \mathbb{R}^{d_v \times n}$ (value representations),
- $A \in \mathbb{R}^{n \times n}$ (attention weights).

The result is:

$$
O \in \mathbb{R}^{d_v \times n}
$$

which is the output of the attention mechanism.

---

### Summary

In terms of matrix operations:

1. Compute query, key, and value matrices:

$$
Q = W_Q X, \quad K = W_K X, \quad V = W_V X
$$

2. Compute attention scores:

$$
S = \frac{K^T Q}{\sqrt{d_k}}
$$

3. Apply softmax to scores to get attention weights:

$$
A = \text{Softmax}(S)
$$

4. Compute the output as a weighted sum of values:

$$
O = V A
$$

## Multi-Head Attention Mechanism

To capture a diverse range of features and relationships in the input data, multi-head attention employs multiple attention "heads" that operate in parallel. Each head focuses on different aspects of the data by independently calculating attention outputs using its own learned weight matrices. This approach allows the model to process and integrate multiple perspectives simultaneously, enabling richer representations of the input.

### Parallel Attention Heads and Matrix Combination

Each of the $h$ attention heads computes its own output matrix:

$$
O_i = V_i A_i \quad \text{for } i = 1, 2, \dots, h,
$$

where $O_i \in \mathbb{R}^{d_v \times n}$ represents the output from the $i$-th attention head. To integrate these outputs, the results from all heads are stacked vertically into a single matrix:

$$
O_{\text{combined}} = 
\begin{bmatrix}
O_1 \\
O_2 \\
\vdots \\
O_h
\end{bmatrix} 
\in \mathbb{R}^{(h \cdot d_v) \times n}.
$$

This stacking operation forms a larger matrix where each attention head contributes a block of rows corresponding to its output. The result is a multi-faceted representation of the input that encapsulates the distinct features learned by each head.

### Dimensionality Constraints for Attention Heads

To ensure that the combined output aligns with the model's overall feature dimension $d$, the relationship between the number of attention heads ($h$), the dimension of each head’s output ($d_v$), and the total model dimension ($d$) must satisfy:

$$
h \cdot d_v = d,
$$

where:
- $d$ is the model’s total feature dimension,
- $h$ is the number of attention heads,
- $d_v$ is the feature dimension of each head.

This constraint ensures that the combined matrix $O_{\text{combined}}$ has a total of $d$ rows, regardless of how many heads are used. As the number of heads increases, the dimension of each head’s output ($d_v$) must decrease proportionally. This tradeoff balances the model's ability to focus on diverse features across heads with the capacity of individual heads to represent detailed patterns.

By stacking attention outputs in this way, multi-head attention integrates diverse insights from the input data while preserving a structured and scalable approach to feature representation.

## Attention Layer

The Transformer architecture, introduced in the paper [*Attention Is All You Need*](https://arxiv.org/html/1706.03762v7) (Vaswani et al., 2017), is built around the multi-head self-attention mechanism, supported by feed-forward layers and add and norm operations. This section focuses on these **support layers**, which, as shown in the architecture diagram below. 
![Transformer architecture](https://arxiv.org/html/1706.03762v7/extracted/1706.03762v7/Figures/ModalNet-21.png)

### First Add and Norm Layer

The first add and norm layer integrates the output of the multi-head attention mechanism back into the network. Multi-head attention can be represented as a function:

$$
\text{MultiHeadAttention}(X)
$$

This function processes the input \( X \) through attention heads and combines their outputs into a single representation.

The add and norm layer then performs the following steps:

1. **Residual Connection**: Adds the original input \( X \) to the attention output:
   $$
   Y = X + \text{MultiHeadAttention}(X)
   $$

2. **Layer Normalization**: [Normalizes](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization) the combined result for stability:
   $$
   Z = \text{LayerNorm}(Y)
   $$

This operation can be written concisely as:

$$
Z = \text{LayerNorm}(X + \text{MultiHeadAttention}(X))
$$

It ensures stable gradients and prepares the output for further processing.


### Feed-Forward Layer

The feed-forward layer complements multi-head attention by applying position-wise transformations (from [Attention Is All You Need](https://arxiv.org/html/1706.03762v7)) to enrich token representations. It includes an intermediate hidden layer with a larger dimension \( d_\text{ffn} \), typically several times larger than the model dimension \( d \). This enhances the model’s capacity to capture complex patterns.

The layer performs ([ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) is the activation function):

1. **Linear Transformation and Activation**:
   $$
   H = \text{ReLU}(Z W_1 + b_1), \quad W_1 \in \mathbb{R}^{d_\text{ffn} \times d}, \, b_1 \in \mathbb{R}^{d_\text{ffn}}
   $$

2. **Second Linear Transformation**:
   $$
   F = H W_2 + b_2, \quad W_2 \in \mathbb{R}^{d \times d_\text{ffn}}, \, b_2 \in \mathbb{R}^d
   $$

The larger intermediate dimension  $d_{ffn}$ allows the layer to model complex interactions before reducing back to $d$. The output is then passed through another add and norm layer for stability.

### Summary of Attention Layer

1. **Multi-Head Attention**:
   $$
   \text{MultiHeadAttention}(X) = \text{Concat}(O_1, \dots, O_h) W_O, \quad O_i = \text{Softmax}\left(\frac{K_i^T Q_i}{\sqrt{d_k}}\right) V_i
   $$

2. **Add and Norm**:
   $$
   Z = \text{LayerNorm}(X + \text{MultiHeadAttention}(X))
   $$

3. **Feed-Forward**:
   $$
   F = \text{ReLU}(Z W_1 + b_1) W_2 + b_2
   $$

4. **Second Add and Norm**:
   $$
   \text{Output}_{\text{Attention Layer}} = \text{LayerNorm}(Z + F)
   $$

The final output, \( \text{Output}_{\text{Attention Layer}} \), represents the result of processing through the attention layer, integrating multi-head attention and feed-forward operations.

---

## Transformer Block

The Transformer architecture, originally introduced for translation in [*Attention Is All You Need*](https://arxiv.org/html/1706.03762v7), consists of \( N \) stacked layers of encoder and decoder blocks. The **encoder** processes input sequences into high-dimensional representations, while the **decoder** generates output sequences token by token.

Frameworks like [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) provide implementations of these components, making it easy to instantiate and customize Transformer blocks in Julia.

---

### Encoder and Decoder Use Cases

The encoder and decoder can be used independently for various tasks:

- **Encoder-Only**: For tasks like classification or sound processing, only the encoder is needed. For example, sound analysis uses the encoder to extract features, simplifying training by avoiding sequence generation.
- **Decoder-Only**: Models like **ChatGPT** and **Bard** use decoders for text generation by attending to previously generated tokens.

---

### Simplifying Training with Encoders

Encoders process input sequences in parallel, avoiding the need for an **attention mask** used in decoders to prevent "looking ahead." This simplification is particularly useful in applications like sound processing and other non-generative tasks.

---

### Example: Instantiating a Transformer Block in Julia

Using the [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) package, you can instantiate a Transformer block as follows:

```julia
using Transformers

# Instantiate a Transformer encoder block
transformer_block = TransformerBlock(
    model_dim = 512,       # Dimension of the model (d)
    num_heads = 8,         # Number of attention heads
    ffn_dim = 2048,        # Feed-forward layer dimension (d_ffn)
    num_layers = 6         # Number of stacked layers (N)
)
```

This creates a Transformer block with an encoder structure, ready to process input sequences for tasks like sound processing or text classification.