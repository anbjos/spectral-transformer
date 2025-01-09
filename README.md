# Spectral Transformer

The **Spectral Transformer** repository demonstrates the versatility of transformer-based AI models in signal processing tasks, with a particular focus on **audio applications** like noise suppression. Unlike traditional transformer inputs consisting of tokenized text, this project adapts modern transformer architectures to work directly with **audio signals** represented as mel-spectrograms. The transformers are customized to accommodate this unique input format, enabling highly effective signal analysis and modification.

A key highlight is the integration of neural networks and digital signal processing (DSP) techniques. The results showcase the synergy of these approaches, achieving precise noise suppression while maintaining the **original signal's phase information**. Notably, this approach avoids the use of [vocoders](https://en.wikipedia.org/wiki/Phase_vocoder) for audio reconstruction, ensuring higher fidelity in the output.

## Results

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

## Transformer

This section presents a textbook description of the Transformer, which may be skipped by those already familiar with the concept. It provides a matrix representation aligned with the Julia format, where inputs are arranged as $inputs \times sequence$, unlike formats used in the [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) paper. Additionally, it lays the groundwork for modifications needed to handle audio data instead of text, with the implementation following this structure.


### Attention Head

The attention mechanism is a foundational concept in modern deep learning architectures, such as the Transformer. It enables models to focus on specific elements of an input sequence when generating an output, capturing the relationships between elements of the sequence. This section provides a step-by-step explanation of how the attention mechanism operates for **a single attention head**, starting with input representation as matrices, followed by transformations into **queries (Q)**, **keys (K)**, and **values (V)**—all of which are *learned* representations rather than direct input data. The similarities between **Q** and **K** are then measured using a dot product, yielding weights that indicate how relevant each position is relative to the current query. These weights are subsequently applied to **V**, allowing the model to focus on and aggregate information from different parts of the sequence. Each step is described in terms of the matrix operations that underlie the mechanism.

#### 1. Input Representation as a Matrix

The input sequence is represented as a matrix:

Let the sequence have $n$ elements, each represented by a vector of dimension $d$. The input sequence to the attention head is then represented as:

$$
X \in \mathbb{R}^{d \times n}
$$

where each column corresponds to the representation of an element in the sequence.

#### 2. Linear Transformations to Obtain Query, Key, and Value

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

#### 3. Attention Score Calculation

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

#### 4. Softmax to Compute Attention Weights

The [softmax](https://en.wikipedia.org/wiki/Softmax_function) function is applied row-wise to the scaled attention scores to produce attention weights, ensuring that each row forms a probability distribution that sums to 1:

$$
A = \text{Softmax}\left(\frac{K^T Q}{\sqrt{d_k}}\right)
$$

For the $i$-th query, the attention weight for the $j$-th key is:

$$
A_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^n \exp(S_{ik})}
$$

This results in the attention weights matrix $A \in \mathbb{R}^{n \times n}$, where each row represents how much each key contributes to the respective query.

#### 5. Weighted Sum Using Values

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
### Explanation Summary

The attention mechanism identifies and amplifies important relationships between sequence elements. For example, if the first and third column in $X$ have high similarity (as measured by the dot product), the attention score $S_{1,3}$ will be large. After applying softmax, this results in a high weight $A_{1,3}$, causing the third column in $V$ to contribute significantly to the first column in $O$.

The weight matrices $W_Q$, $W_K$, and $W_V$ are learned during training and independently transform each input column into corresponding columns of queries, keys, and values. This enables the model to capture the necessary relationships for effective task performance.


## Multi-Head Attention Mechanism

To capture a diverse range of features and relationships in the input data, multi-head attention employs multiple attention "heads" that operate in parallel. Each head focuses on different aspects of the data by independently calculating attention outputs using its own learned weight matrices. This approach allows the model to process and integrate multiple perspectives simultaneously, enabling richer representations of the input.

### Parallel Attention Heads and Matrix Combination

Each of the $h$ attention heads computes its own output matrix:

$$
O_i = V_i A_i \quad \text{for } i = 1, 2, \dots, h,
$$

where $O_i \in \mathbb{R}^{d_v \times n}$ represents the output from the $i$-th attention head. To integrate these outputs, the results from all heads are stacked vertically into a single matrix:

$$
O_{\text{MultiHead}} = 
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

This constraint ensures that the combined matrix $O_{\text{MultiHead}}$ has a total of $d$ rows, regardless of how many heads are used. As the number of heads increases, the dimension of each head’s output ($d_v$) must decrease proportionally. This tradeoff balances the model's ability to focus on diverse features across heads with the capacity of individual heads to represent detailed patterns.

By stacking attention outputs in this way, multi-head attention integrates diverse insights from the input data while preserving a structured and scalable approach to feature representation.

## Attention Layer

The Transformer architecture, introduced in the paper [*Attention Is All You Need*](https://arxiv.org/html/1706.03762v7) (Vaswani et al., 2017), is built around the multi-head self-attention mechanism, supported by feed-forward layers and add and norm operations. This section focuses on these **support layers**, which, as shown in the architecture diagram below. 
![Transformer architecture](https://arxiv.org/html/1706.03762v7/extracted/1706.03762v7/Figures/ModalNet-21.png)

### First Add and Norm Layer

The first add and norm layer integrates the output of the multi-head attention mechanism back into the network. Multi-head attention can be represented as a function:

$$
Y=\text{MultiHeadAttention}(X)
$$

Here $X,Y,Z$ represent inputs to functions and

$$
X\in \mathbb{R}^{d \times n},Y\in \mathbb{R}^{d \times n}, Z\in \mathbb{R}^{d \times n}
$$

This function processes the input \( X \) through attention heads and combines their outputs into a single representation.

The add and norm layer then performs the following steps:

1. **Residual Connection**: Adds the original input \( X \) to the attention output:

$$
Y = X + \text{MultiHeadAttention}(X)
$$

2. **Layer Normalization**: [Normalizes](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization) the combined result for stability:

$$
Y = \text{LayerNorm}(X)
$$

Combining these into one function provide:

$$
FirstNormalization(X) = \text{LayerNorm}(X + \text{MultiHeadAttention}(X))
$$

It ensures stable gradients and prepares the output for further processing.

### Feed-Forward Layer

The feed-forward layer processes the normalized output $Z$ with two linear transformations and a ReLU activation:

1. **Linear Transformation and Activation**:

$$
Y = \text{ReLU}(W_1 X + b_1)
$$

where $W_1 \in \mathbb{R}^{d_\text{ffn} \times d}$ and $b_1 \in \mathbb{R}^{d_\text{ffn}}$. Typically, $d_\text{ffn}$ is set to **4 times** the input dimension $d$, allowing the layer to model more complex interactions.

2. **Second Linear Transformation**:

$$
Y = W_2 X + b_2
$$

where $W_2 \in \mathbb{R}^{d \times d_\text{ffn}}$ and $b_2 \in \mathbb{R}^d$. This transformation reduces the dimension back to $d$, ensuring the output size remains consistent with the original input.

Combining these provides:

$$
FeedForwardLayer(X) = W_2 (\text{ReLU}(W_1 X + b_1)) + b_2
$$

**Why increase and then decrease the dimension?**  
Increasing the dimension to $d_\text{ffn}$ (typically $4d$) enables the model to capture richer and more complex features in a higher-dimensional space. Reducing the dimension back to $d$ ensures the output remains manageable and compatible with the rest of the architecture, while still benefiting from the complex intermediate representations.

Finally, the result is integrated using another add and norm layer. The complete chain can be written:

$$
Z=FirstNormalization(X)\\
Y = LayerNorm(FeedForwardLayer(Z)+Z)
$$

---

## Transformer Block

The system considered here is an **encoder**, composed of $N$ stacked **Transformer attention layers**, each integrating multi-head attention, feed-forward layers, and add and norm operations. These layers process input sequences into high-dimensional representations, making the encoder suitable for tasks like classification, sound processing, or summarization.

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

$$\text{Embedding}(U) = W_{\text{emb}} U,$$

where $W_{\text{emb}} \in \mathbb{R}^{d \times d_U}$. Thus,

$$\text{Embedding}(U) \in \mathbb{R}^{d \times n}.$$

Each column of $\text{Embedding}(U)$ represents a sequence element, enabling the Transformer to process the audio data effectively.

### Output Projection

After the Transformer processes the embedded features, we convert the output back to the original audio feature space through an **Output Projection** step, implemented as another linear layer:

$$\text{OutputProjection}(Y) = W_{\text{out}} Y,$$

where $W_{\text{out}} \in \mathbb{R}^{d_U \times d}$ and $Y \in \mathbb{R}^{d \times n}$. Consequently,

$$\text{OutputProjection}(Y) \in \mathbb{R}^{d_U \times n}.$$

### Independent Learned Transformations

Both the **Input Embedding** matrix $W_{\text{emb}}$ and the **Output Projection** matrix $W_{\text{out}}$ are **learned independently** during training. There is **no enforced relationship** between them, allowing each to optimize its transformation for its specific role:

- **Input Embedding ($W_{\text{emb}}$):** Projects raw audio features into the Transformer's embedding space.
- **Output Projection ($W_{\text{out}}$):** Maps the Transformer's output back to the original audio feature space.


---

## Positional Encoding

The positional encoding $\text{PE}(i, j)$ is defined for each embedding dimension $i$ and token position $j$ as:

$$
\text{PE}(i, j) =
\begin{cases}
\sin\left(\frac{j}{10000^{i/d}}\right), & \text{if } i \text{ is even}, \\
\cos\left(\frac{j}{10000^{(i-1)/d}}\right), & \text{if } i \text{ is odd}.
\end{cases}
$$

Here:
- $j$ is the token's position in the sequence.
- $i$ is the embedding dimension index.
- $d$ is the embedding dimensionality.

This creates a matrix $\mathbf{P}$ where each column represents a token’s positional encoding. The matrix is visualized below, showing that the embedding for the first position oscillates with a very high frequency, while the frequency decreases for larger positions.

![positional encoding](figures/position.png)

If we multiply the embeddings of two different positions and plot the resulting $sequence \times sequence$ matrix, the following pattern emerges:

![distance](figures/distance.png)

The raw dot product between two positional encodings defines a function that measures the relative distance between the two positions. Since the attention mechanism involves a dot product between $K^T$ and $Q$, it *can* capture the relative distance between inputs in the sequence. The term *can* is used because this process also involves multiplication with learned weights, allowing the model to adjust these weights during training to capture embedded data, relative or absolute position, or a combination of both.

---

### Dot product of positional Embeddings

#### 1. Column Construction (Position as Column Index)

To understand the mechanism behind this, assume the positional encoding matrix $\mathbf{P}$ has:
- **$d$ rows** (each row corresponds to a dimension of the embedding),
- **$n$ columns** (each column corresponds to a position $j$ in the sequence).

For position $j$, the column $\mathbf{p}\_j$ looks like

$$\mathbf{p}\_j = \bigl[\sin\bigl(\tfrac{j}{\alpha\_0}\bigr),\,\cos\bigl(\tfrac{j}{\alpha\_0}\bigr),\,\sin\bigl(\tfrac{j}{\alpha\_1}\bigr),\,\cos\bigl(\tfrac{j}{\alpha\_1}\bigr),\,\dots\bigr]^\top$$

where each $\alpha\_k$ (often $10000^{k/d}$) controls the wavelength of the $k$-th sine/cosine pair.

#### 2. Dot Product Involves Sine-Cosine Products

To compute the dot product between two columns $\mathbf{p}\_j$ and $\mathbf{p}\_{j'}$ (positions $j$ and $j'$), we pairwise multiply elements and sum them:

$$\mathbf{p}\_j^\top\mathbf{p}\_{j'} = \sum_{k\in\text{(sine/cos pairs)}}\bigl[\sin\bigl(\tfrac{j}{\alpha\_k}\bigr)\sin\bigl(\tfrac{j'}{\alpha\_k}\bigr) + \cos\bigl(\tfrac{j}{\alpha\_k}\bigr)\cos\bigl(\tfrac{j'}{\alpha\_k}\bigr)\bigr].$$

Using the trigonometric identity

$$\sin(A)\sin(B)+\cos(A)\cos(B)=\cos(A-B),$$

each sine-cosine pair becomes

$$\cos\bigl(\tfrac{j}{\alpha\_k}-\tfrac{j'}{\alpha\_k}\bigr)=\cos\Bigl(\tfrac{j-j'}{\alpha\_k}\Bigr).$$

#### 3. Dependence on $(j-j')$

Summing across all frequencies $\alpha\_k$ yields terms of the form $\cos\bigl(\tfrac{j-j'}{\alpha\_k}\bigr)$. Hence, the dot product depends on the difference $(j-j')$:

$$\mathbf{p}\_j \mathbf{p}\_{j'} = \sum_{k}\cos\Bigl(\frac{j-j'}{\alpha\_k}\Bigr).$$

Because this expression depends only on $(j-j')$ and **not** on $j$ or $j'$ separately, it encodes the **relative distance** between these two positions in the sequence.

---

## Masking in the Encoder

### Purpose of Masking

1. Handling Variable Sequence Lengths:  
   Masks prevent the encoder from attending to padding tokens, ensuring that attention mechanisms focus only on meaningful data.

2. Enforcing Causality:  
   Masks restrict attention to previous positions in the sequence, maintaining the temporal order and preventing the model from accessing future information during training.

### Current Application Context

In our **noise suppression** application, all input sequences have a **constant length**, eliminating the immediate need for masking. However, we retain the masking mechanism to support potential future enhancements, such as:

- **Variable-Length Inputs:** Allowing the model to handle audio samples of different duration's without structural changes.
- **Enhanced Feature Integration:** Facilitating the inclusion of additional features that may require selective attention controls.

By maintaining the masking infrastructure, we ensure that the model remains flexible and adaptable to evolving requirements, even though masking is not strictly necessary for the current fixed-length sequence setup.

## Masking in the Encoder

In Transformer architectures, **masking** controls the flow of information during training. Within the **encoder**, masks primarily address varying sequence lengths and enforce causality, similar to their role in the **decoder**.

### Purpose of Masking

1. **Handling Variable Sequence Lengths:**  
   Masks prevent the encoder from attending to padding tokens, ensuring that attention mechanisms focus only on meaningful data.

2. **Enforcing Causality:**  
   Masks restrict attention to previous positions in the sequence, maintaining the temporal order and preventing the model from accessing future information during training.

### Current Application Context

In our **noise suppression** application, all input sequences have a **constant length**, eliminating the immediate need for masking. However, we retain the masking mechanism to support potential future enhancements, such as:

- **Variable-Length Inputs:** Allowing the model to handle audio samples of different duration's without structural changes.
- **Enhanced Feature Integration:** Facilitating the inclusion of additional features that may require selective attention controls.

By maintaining the masking infrastructure, we ensure that the model remains flexible and adaptable to evolving requirements, even though masking is not strictly necessary for the current fixed-length sequence setup.

---

## Model Summary

The **Spectral Transformer** suppresses audio noise through the following pipeline:

1. **Embedding and Positional Encoding**
   
   $$
   X=\text{PositionalEncoding}(\text{Embedding}(U))
   $$

2. **Transformer Processing and Output Projection**
   
   $$
   Y=\text{OutputProjection}(\text{Transformer}(X))
   $$

**Key Points:**

- $W_{\text{emb}}$ and $W_{\text{out}}$ are **independently learned**.
- Masking is included to support future enhancements, such as handling variable-length inputs.

## Audio Signal Processing

Before feeding raw audio signals into the **Spectral Transformer**, they undergo a carefully designed signal processing pipeline. Each step in this pipeline is tailored to enhance the quality of the data and improve the neural network's performance. By addressing different aspects of the audio signal, these preprocessing steps ensure that the transformer operates on meaningful, well-structured data. The following sections describe each step and its purpose in preparing the signal for effective noise suppression.

---

### Short-Time Fourier Transform

The first step involves applying the [Short-Time Fourier Transform (STFT)](https://en.wikipedia.org/wiki/Short-time_Fourier_transform) to convert the audio signal from the time domain to a sequence of Short Fourier Transforms. This is done using an FFT length of $d_U$ and a modified [Hanning window](https://en.wikipedia.org/wiki/Hann_function):

$$w(k) = \sin^2\left(\frac{\pi k}{d_U}\right), \quad k \in [1 \dots d_U]$$

Although the first element of this window is not zero, it is still effective due to its smooth tapering, which minimizes spectral leakage. Overlapping windows (with a step size of $\frac{d_U}{2}$) ensure continuity, as the peak value (1) of one window aligns with the zero value of the previous, and the sum of overlapping window values is always 1. This simplifies signal reconstruction (see **Signal Reconstruction** for details).

The resulting STFT is given by:

$$X_{\mathbb{C}} = \text{stft}(u, w, \text{step} = \frac{d_U}{2})$$

Here, the subscript $\mathbb{C}$ indicates that the resulting matrix is complex, containing both amplitude and phase information in the frequency domain. For a real input signal, the STFT returns the [DC component](https://en.wikipedia.org/wiki/DC_bias), positive frequencies, and the [Nyquist frequency](https://en.wikipedia.org/wiki/Nyquist_frequency), since negative frequencies are the complex conjugates of positive ones.

---

### Spectrogram

Since the transformer operates in the power domain, $X_{\mathbb{C}}$ is converted to a spectrogram by taking the squared magnitude of each element:

$$X_{P} = \text{spectrogram}(X_{\mathbb{C}})$$

Some scaling is applied during this process, depending on the data dimensions, but this is not critical due to subsequent normalization steps.

---

### Mel Representation

Human perception is more sensitive to variations in lower frequencies, making it possible to reduce the dimensionality of the data using a [Mel-frequency scale](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). In this application, only the Mel filtering step is used, omitting the [Discrete Cosine Transform (DCT)](https://en.wikipedia.org/wiki/Discrete_cosine_transform).

This step involves a simple linear transformation:

$$X_M = dB(M \times X_P), \quad M \in \mathbb{R}^{m \times d_U}, \quad X_M \in \mathbb{R}^{m \times n}$$

A code snippet for constructing the Mel filter matrix $M$ can be found at [Practical Cryptography](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) and can be easily adapted to Julia.

The function [dB](https://en.wikipedia.org/wiki/Decibel) convert the power level to a logarithmic scale by applying $dB(P)=10 \times log_{10}(P)$ to each element in the matrix.

---

### Power clamping

One issue with the logarithmic representation is that low power levels, that really do not contribute the the audio signal, is represented as large negative number. This can affect the training, since the network tries to learn these instead of the part of the signal that actually mean something. To deal with this, the signal is clamped to be above a threshold, that is defined maximum power level for the signal or the noise, attenuated by 60 dB.

---

### Whitening

The final step before feeding the data to the model is **whitening**, which normalizes the signal by setting its mean to zero and standard deviation to one. This ensures consistent input statistics, improving training stability and convergence. The stored bias and scaling factors are later used to reverse the transformation on the model’s output, allowing correct reconstruction of the enhanced audio.

Whitening helps reduce bias from background noise and enhances model generalization by ensuring the network focuses on patterns rather than variations in magnitude. This step is crucial for reliable performance across diverse audio signals.

$$U=whiten(clamp(X_M))$$

