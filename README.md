# Spectral Transformer

The **Spectral Transformer** repository demonstrates the versatility of transformer-based AI models in signal processing tasks, with a particular focus on **audio applications** like noise suppression. Unlike traditional transformer inputs consisting of tokenized text, this project adapts modern transformer architectures to work directly with **audio signals** represented as mel-spectrograms. The transformers are customized to accommodate this unique input format, enabling highly effective signal analysis and modification.

A key highlight is the integration of neural networks and digital signal processing (DSP) techniques. The results showcase the synergy of these approaches, achieving precise noise suppression while maintaining the **original signal's phase information**. Notably, this approach avoids the use of [vocoders](https://en.wikipedia.org/wiki/Phase_vocoder) for audio reconstruction, ensuring higher fidelity in the output.

Even with a relatively simple transformer network, excellent performance can be achieved in real-world scenarios, as illustrated by the following example. Using the challenging task of separating human speech from background noise (e.g., frog sounds), the network demonstrates its robustness. Specifically:

1. **Signal with Noise** (Figure 1): This represents the input noisy speech signal, containing both human speech and frog sounds.

![Signal with noise](figures/with_noise.png)
![Listen to signal with noise](https://github.com/anbjos/spectral-transformer/blob/main/figures/with_noise.wav) (Choose "Download Raw" to hear the audio)

2. **Signal without Noise** (Figure 2): This is the clean speech signal, used as a reference for comparison.

![Signal without noise](figures/signal.png)
![Listen to signal](https://github.com/anbjos/spectral-transformer/blob/main/figures/signal.wav) (Choose "Download Raw" to hear the audio)

3. **Cleaned Signal with Noise** (Figure 3): This is the output of the network, showing its ability to effectively suppress noise and isolate the speech signal, even on out-of-sample data.

![Cleaned signal with noise](figures/result.png)
![Listen to result](https://github.com/anbjos/spectral-transformer/blob/main/figures/result.wav) (Choose "Download Raw" to hear the audio)


## Attention Mechanism
Understanding the [attention mechanism](https://arxiv.org/html/1706.03762v7) is crucial for adapting deep learning architectures to various input formats. While most explanations utilize a sequence × embedding input format, this discussion emphasizes the embedding × sequence format as implemented in [Julia](https://julialang.org/). Additionally, this documentation aims to provide a comprehensive overview of the concepts that may initially seem challenging, breaking them down in a clear and accessible manner. By doing so, it not only facilitates modifications to input/output processing but also ensures the provided code is both intuitive and adaptable.

The attention mechanism is a foundational concept in modern deep learning architectures, such as the Transformer. It enables models to focus on specific elements of an input sequence when generating an output, capturing the relationships between elements of the sequence. This section provides a step-by-step explanation of how the attention mechanism operates for **a single attention head**, starting with input representation as matrices, followed by transformations into **queries (Q)**, **keys (K)**, and **values (V)**—all of which are *learned* representations rather than direct input data. The similarities between **Q** and **K** are then measured using a dot product, yielding weights that indicate how relevant each position is relative to the current query. These weights are subsequently applied to **V**, allowing the model to focus on and aggregate information from different parts of the sequence. Each step is described in terms of the matrix operations that underlie the mechanism.

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

Each element of the input sequence is transformed into three separate representations: **query ($Q$)**, **key ($K$)**, and **value ($V$)** using weight matrices:

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
- $d_k$ is the dimension of the query and key vectors.  
- $d_v$ is the dimension of the value vectors.

The weight matrices $W_Q$, $W_K$, and $W_V$ are trained during the learning process, with each attention head independently learning its own set of weights.

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

The feed-forward layer processes the normalized output $Z$ with two linear transformations and a ReLU activation:

1. **Linear Transformation and Activation**:

$$
H = \text{ReLU}(W_1 Z + b_1)
$$

where $W_1 \in \mathbb{R}^{d_\text{ffn} \times d}$ and $b_1 \in \mathbb{R}^{d_\text{ffn}}$. Typically, $d_\text{ffn}$ is set to **4 times** the input dimension $d$, allowing the layer to model more complex interactions.

2. **Second Linear Transformation**:

$$
F = W_2 H + b_2
$$

where $W_2 \in \mathbb{R}^{d \times d_\text{ffn}}$ and $b_2 \in \mathbb{R}^d$. This transformation reduces the dimension back to $d$, ensuring the output size remains consistent with the original input.

Finally, the result is integrated using another add and norm layer:

$$
\text{Output}_{\text{Attention Layer}} = \text{LayerNorm}(Z + F)
$$

**Why increase and then decrease the dimension?**  
Increasing the dimension to $d_\text{ffn}$ (typically $4d$) enables the model to capture richer and more complex features in a higher-dimensional space. Reducing the dimension back to $d$ ensures the output remains manageable and compatible with the rest of the architecture, while still benefiting from the complex intermediate representations.

---

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
F = \text{ReLU}(W_1 Z + b_1) W_2 + b_2
$$

4. **Second Add and Norm**:

$$
\text{Output}_{\text{Attention Layer}} = \text{LayerNorm}(X + F)
$$

The final output, $Output_{\text{Attention Layer}}$, represents the result of processing through the attention layer, integrating multi-head attention and feed-forward operations.

---

## Transformer Block

The system considered here is an **encoder**, composed of $N$ stacked **Transformer attention layers**, each integrating multi-head attention, feed-forward layers, and add and norm operations. These layers process input sequences into high-dimensional representations, making the encoder suitable for tasks like classification, sound processing, or summarization.

---

### Encoder and Decoder in the Original Transformer

The original Transformer architecture, as introduced in [*Attention Is All You Need*](https://arxiv.org/html/1706.03762v7), included both an encoder and a decoder:

1. **Encoder**:
   - Processes input sequences into high-dimensional representations.
   - Fully parallelizable, making it efficient for non-generative tasks.

2. **Decoder**:
   - Generates output sequences token by token, attending to both encoder outputs and previously generated tokens.
   - Requires an **attention mask** during training to prevent "looking ahead."

While encoder-decoder configurations are used for tasks like translation and summarization (e.g., **BART**), encoder-only systems simplify training and are ideal for tasks such as classification and sound processing. Decoder-only models, like **ChatGPT**, focus on text generation.

---

### Example: Instantiating an Encoder Block in Julia

Using [Transformers.jl](https://github.com/chengchingwen/Transformers.jl):

```julia
using Transformers

# Instantiate an encoder block
transformer_block = TransformerBlock(
    model_dim = 512,       # Model dimension (d)
    num_heads = 8,         # Attention heads
    ffn_dim = 2048,        # Feed-forward dimension (d_ffn)
    num_layers = 6         # Number of layers (N)
)
```

Frameworks like [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) provide modular implementations, enabling easy customization of encoder blocks. This example demonstrates how to set up a stack of \( N \) attention layers for encoder-based tasks.

---

## Input and Output Embedding for Audio in the Frequency Domain

In our **noise suppression** application, we operate on **audio signals in the frequency domain** using a matrix $U \in \mathbb{R}^{d_U \times n}$, where:

- $d_U$ is the **feature dimension** (e.g., mel-frequency bins),
- $n$ is the **sequence length** (e.g., number of time frames).

### Input Embedding

Before passing $U$ into the Transformer, we apply an **Input Embedding** step that projects $U$ to the model’s embedding dimension $d$ using a linear layer:

$$
\text{Embedding}(U) = W_{\text{emb}} \, U,
$$

where $W_{\text{emb}} \in \mathbb{R}^{d \times d_U}$. Thus,

$$
\text{Embedding}(U) \in \mathbb{R}^{d \times n}.
$$

Each column of $\text{Embedding}(U)$ represents a sequence element, enabling the Transformer to process the audio data effectively.

### Output Projection

After the Transformer processes the embedded features, we convert the output back to the original audio feature space through an **Output Projection** step, implemented as another linear layer:

$$
\text{OutputProjection}(Y) = W_{\text{out}} \, Y,
$$

where $W_{\text{out}} \in \mathbb{R}^{d_U \times d}$ and $Y \in \mathbb{R}^{d \times n}$. Consequently,

$$
\text{OutputProjection}(Y) \in \mathbb{R}^{d_U \times n}.
$$

### Independent Learned Transformations

Both the **Input Embedding** matrix $W_{\text{emb}}$ and the **Output Projection** matrix $W_{\text{out}}$ are **learned independently** during training. There is **no enforced relationship** between them, allowing each to optimize its transformation for its specific role:

- **Input Embedding ($W_{\text{emb}}$):** Projects raw audio features into the Transformer's embedding space.
- **Output Projection ($W_{\text{out}}$):** Maps the Transformer's output back to the original audio feature space.

### Summary

In this noise suppression system:

- **Input Embedding:** Transforms frequency-domain audio features into the Transformer's embedding space via a learned linear layer.
- **Output Projection:** Converts the Transformer's output back to the original audio feature space through a separate learned linear layer.

Both embeddings are **independently learned**, ensuring optimal transformations for noise suppression without imposed constraints.

---

## Positional Encoding

The Transformer views its input matrix $\mathbf{U}$ as consisting of **columns** that represent elements in a sequence:

$$
\mathbf{U} = \bigl[\mathbf{u}_1,\;\mathbf{u}_2,\;\dots,\;\mathbf{u}_n\bigr].
$$

By default, the dot-product self-attention mechanism has **no inherent sense** of the order of these columns within the sequence. To introduce this ordering information, we define the positional encoding as a function:

$$
\text{PositionalEncoding}(\mathbf{U}) = \mathbf{U} + \mathbf{P},
$$

where $\mathbf{P}$ is constructed from **sinusoidal terms** that encode each column’s position. This enables the Transformer to learn relationships based on both **content** and **relative positions** within the sequence.

---

## 1. What It Does

1. **Inject Ordering**  
   Without positional encoding, self-attention treats all columns $\mathbf{u}_i$ equally, ignoring their sequence order (e.g., first, second, etc.). By adding position-dependent encodings, each column “knows” its position in the sequence.

2. **Enable Relative Distance Awareness**  
   The **similarity** between two columns $\mathbf{u}_i$ and $\mathbf{u}_j$ can now depend on their **relative distance** $|i - j|$. This allows the model to learn patterns such as “focus on elements 1 step away” or “focus on elements 5 steps away,” regardless of their absolute positions in the sequence.

3. **Support Generalization**  
   Because the positional encodings use **sine and cosine functions** at multiple frequencies, the Transformer can apply the **same** relative-distance-based attention strategies to any pair of positions $i, j$, even in sequences longer or shorter than those seen during training.

---

## 2. How It Works

### 2.1 Sinusoidal Vectors per Position

For each position $i = 1,\dots,n$ (i.e., column index) and each dimension index $j$, the **even** entries use sine functions and the **odd** entries use cosine functions with varying wavelengths:

$$
\mathbf{PE}(i,\,2j) = \sin\!\Bigl(\tfrac{i}{10000^{\,2j/d}}\Bigr), \quad \mathbf{PE}(i,\,2j+1) = \cos\!\Bigl(\tfrac{i}{10000^{\,2j/d}}\Bigr).
$$

Here, $d$ is the embedding dimension (matching the dimensionality of each $\mathbf{u}_i$), and $j$ ranges from $0$ to $\tfrac{d}{2} - 1$. This setup ensures a **range of frequencies**, allowing the model to capture both fine-grained and broad positional relationships.

### 2.2 Adding PE to the Input Matrix

We construct a positional encoding matrix $\mathbf{P} \in \mathbb{R}^{d \times n}$ by stacking all $\mathbf{PE}(i)$ as columns:

$$
\mathbf{P} = \bigl[\mathbf{PE}(1),\;\mathbf{PE}(2),\;\dots,\;\mathbf{PE}(n)\bigr].
$$

The position-enriched input is then:

$$
\mathbf{U}' = \mathbf{U} + \mathbf{P}.
$$

Each updated column $\mathbf{u}_i' = \mathbf{u}_i + \mathbf{PE}(i)$ now contains **positional** information alongside the original **content**.

### 2.3 Position-Aware Queries, Keys, and Values

Within the Transformer’s self-attention mechanism, the columns of $\mathbf{U}'$ are linearly projected to form **queries** ($\mathbf{Q}$), **keys** ($\mathbf{K}$), and **values** ($\mathbf{V}$):

$$
\mathbf{Q} = \mathbf{W}_Q\,\mathbf{U}',\quad \mathbf{K} = \mathbf{W}_K\,\mathbf{U}',\quad \mathbf{V} = \mathbf{W}_V\,\mathbf{U}',
$$

where $\mathbf{W}_Q$, $\mathbf{W}_K$, and $\mathbf{W}_V$ are learned parameter matrices. Each query $\mathbf{q}_i$ and key $\mathbf{k}_j$ thus incorporate both **content** (from $\mathbf{u}_i$ and $\mathbf{u}_j$) and **positional information** (from $\mathbf{PE}(i)$ and $\mathbf{PE}(j)$).

### 2.4 Relative Distances via Multiplicative Interactions

When computing the attention score between the $i$-th and $j$-th columns, the Transformer uses the dot product of their corresponding queries and keys:

$$
\alpha_{ij} = \frac{\mathbf{q}_i^\top\,\mathbf{k}_j}{\sqrt{d_k}} = \frac{(\mathbf{W}_Q(\mathbf{u}_i + \mathbf{PE}(i)))^\top\, (\mathbf{W}_K(\mathbf{u}_j + \mathbf{PE}(j)))}{\sqrt{d_k}}.
$$

Here’s how **relative distance** emerges through **multiplicative interactions**:

- **Phase-Based Multiplicative Interaction encodes relative distance**: The positional encodings $\mathbf{PE}(i)$ and $\mathbf{PE}(j)$ consist of sine and cosine functions with different frequencies. When $\mathbf{q}_i$ and $\mathbf{k}_j$ are multiplied via the dot product, the interaction between their sinusoidal components encodes the **relative distance** $i - j$.

- **Learning Distance-Based Patterns**: Through training, the model learns to associate specific patterns in these phase interactions with meaningful relative distances, enabling it to attend selectively based on how far apart elements are in the sequence.

### 2.5 Generalization Across Positions

The use of multiple frequencies in the sinusoidal positional encodings allows the Transformer to **reuse** the same attention strategies for any pair of positions $i, j$ that share a given relative offset $i - j$. This means the model can **generalize** its learned attention patterns to new sequence lengths and positions, even those not encountered during training.

---

### Conclusion

By defining

$$
\text{PositionalEncoding}(\mathbf{U}) = \mathbf{U} + \mathbf{P},
$$

and constructing $\mathbf{P}$ from **sinusoidal** positional vectors, the Transformer effectively incorporates **relative distance** and **ordering** into its self-attention mechanism. The **multiplicative interactions** within the dot product of queries and keys enable the model to discern and leverage **relative positional relationships**, facilitating robust learning and generalization across diverse and varying sequence lengths.

---

## Masking in the Encoder

In Transformer architectures, **masking** controls the flow of information during training. Within the **encoder**, masks primarily address varying sequence lengths and enforce causality, similar to their role in the **decoder**.

### Purpose of Masking

1. **Handling Variable Sequence Lengths:**  
   Masks prevent the encoder from attending to padding tokens, ensuring that attention mechanisms focus only on meaningful data.

2. **Enforcing Causality:**  
   Masks restrict attention to previous positions in the sequence, maintaining the temporal order and preventing the model from accessing future information during training.

### Current Application Context

In our **noise suppression** application, all input sequences have a **constant length**, eliminating the immediate need for masking. However, we retain the masking mechanism to support potential future enhancements, such as:

- **Variable-Length Inputs:** Allowing the model to handle audio samples of different durations without structural changes.
- **Enhanced Feature Integration:** Facilitating the inclusion of additional features that may require selective attention controls.

By maintaining the masking infrastructure, we ensure that the model remains flexible and adaptable to evolving requirements, even though masking is not strictly necessary for the current fixed-length sequence setup.

Certainly! Below is a **concise summary** of the processing pipeline for the **Spectral Transformer**. This overview outlines the key transformation steps from the raw input to the enhanced audio output, highlighting the roles of embedding, positional encoding, the Transformer model, and output projection. Additionally, it emphasizes that both embedding and projection layers are **independently learned** and that masking is maintained for future flexibility.

---

## Model Pipeline Overview

The **Spectral Transformer** processes audio signals through the following steps:

1. **Input Representation ($U$):**

   The raw audio signal is represented in the frequency domain as a mel-spectrogram matrix:

   $$
   U \in \mathbb{R}^{d_U \times n}
   $$

   where:
   - $d_U$ is the **feature dimension** (e.g., mel-frequency bins),
   - $n$ is the **sequence length** (e.g., number of time frames).

2. **Input Embedding:**

   The input matrix $U$ is projected into the Transformer's embedding space using a learned linear transformation:

   $$
   \text{Embedding}(U) = W_{\text{emb}} \, U
   $$

   where $W_{\text{emb}} \in \mathbb{R}^{d \times d_U}$. Thus,

   $$
   \text{Embedding}(U) \in \mathbb{R}^{d \times n}
   $$

3. **Positional Encoding:**

   Positional information is added to the embedded input to incorporate the sequence order:

   $$
   X = \text{PositionalEncoding}(\text{Embedding}(U)) = \text{Embedding}(U) + \mathbf{P}
   $$

   where $\mathbf{P} \in \mathbb{R}^{d \times n}$ contains **sinusoidal positional encodings**.

4. **Transformer Processing:**

   The position-enriched input $X$ is processed through the Transformer model:

   $$
   Y = \text{Transformer}(X)
   $$

   where $Y \in \mathbb{R}^{d \times n}$ represents the high-level, context-aware representations.

5. **Output Projection:**

   The Transformer's output $Y$ is projected back to the original audio feature space using another learned linear transformation:

   $$
   \text{OutputProjection}(Y) = W_{\text{out}} \, Y
   $$

   where $W_{\text{out}} \in \mathbb{R}^{d_U \times d}$. Consequently,

   $$
   \text{OutputProjection}(Y) \in \mathbb{R}^{d_U \times n}
   $$

### Key Points:

- **Independent Learned Transformations:**  
  Both the **Input Embedding** matrix $W_{\text{emb}}$ and the **Output Projection** matrix $W_{\text{out}}$ are **learned independently** during training. There is **no enforced relationship** between them, allowing each to optimize its transformation for its specific role.

- **Masking Mechanism:**  
  Although masking is **not strictly necessary** for this application due to the **constant sequence length**, the masking infrastructure is **maintained** to support potential future modifications, such as handling variable-length inputs or integrating additional features that may require selective attention controls.

### Processing Pipeline Diagram

```
Input Audio (U)
       |
       v
Input Embedding (Embedding(U) = W_emb * U)
       |
       v
Positional Encoding (X = Embedding(U) + P)
       |
       v
Transformer (Y = Transformer(X))
       |
       v
Output Projection (OutputProjection(Y) = W_out * Y)
       |
       v
Enhanced Audio Output
```
