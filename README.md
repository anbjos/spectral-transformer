# spectral-transformer
A demonstration of transformer-based AI models showcasing their potential in signal processing tasks, such as noise suppression. Highlighted examples include separating human speech and frog sounds, illustrating the versatility and power of modern transformer architectures in audio applications.

## Attention Mechanism

The [attention mechanism](https://arxiv.org/html/1706.03762v7) is a foundational concept in modern deep learning architectures, such as the Transformer. It enables models to focus on specific elements of an input sequence when generating an output, effectively capturing dependencies regardless of sequence length. This section provides a step-by-step explanation of how the attention mechanism operates for **a single attention head**, starting with input representation as matrices, followed by transformations into queries, keys, and values, and concluding with the computation of attention scores and outputs. Each step is described in terms of the matrix operations that underlie the mechanism.

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

These weight matrices $W_Q$, $W_K$, and $W_V$ are learned during training.

After the transformations:
- $Q \in \mathbb{R}^{d_k \times n}$,
- $K \in \mathbb{R}^{d_k \times n}$,
- $V \in \mathbb{R}^{d_v \times n}$.

---

### 3. **Attention Score Calculation**

The **attention mechanism** determines the relevance of each element in the sequence to every other element. This is done by computing a similarity score between queries and keys using a **dot product**:

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

The softmax function is applied row-wise to the scaled attention scores to produce attention weights, ensuring that each row forms a probability distribution that sums to 1:

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


