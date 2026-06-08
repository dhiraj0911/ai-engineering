# How LLMs Work — Notes

## Pipeline Overview

```
Input Text
    ↓
Tokenization
    ↓
Embeddings
    ↓
Transformer Layers
    ↓
Output Logits
    ↓
Probabilities
    ↓
Next Token
```

---

## 1. Tokenization

Split raw text into tokens, then map each token to an integer ID.

**Example:** `"I love pizza"`

```
Tokens  → ["I", " love", " pizza"]
IDs     → [40, 3021, 8421]
```

---

## 2. Embeddings

Each token ID maps to a dense vector — a richer numerical representation of meaning.

```python
token_id  = 8421   # "pizza"
embedding = [0.24, -1.12, 0.83, ...]
```

---

## 3. Transformer Layers

### i. Multi-Head Self-Attention

Each token asks: *"Which other tokens should I pay attention to?"*

**Example:** `"Animal didn't cross the road, because it was tired"`

- `"it"` attends strongly to `"Animal"`, weakly to `"road"`
- Lets the model understand relationships across thousands of tokens

### ii. Feed-Forward Network (FFN)

- Applied after attention
- Acts as a **key-value memory store** for factual knowledge

---

## 4. Next Token Prediction

The model outputs a probability distribution over all possible next tokens and picks the highest one. Generation is autoregressive:

```
Prompt → [token_1] → [token_1, token_2] → [token_1, token_2, token_3] → ...
```

---

## 5. Sampling Strategies

| Parameter       | Description |
|----------------|-------------|
| `temperature`  | Higher → more creative/random; Lower → more deterministic |
| `top_p`        | Nucleus sampling — sample from top tokens summing to probability `p` |
| `top_k`        | Sample from the top-`k` most probable tokens |

---

## 6. Context Window

Everything that fits inside the model's attention at once:

```
[System Prompt] + [Retrieved Chunks] + [Chat History] + [User Query] ≤ Context Limit
```

---

## Fine-Tuning vs RAG

| | What changes |
|---|---|
| **Fine-tuning** | *How* the model thinks — tone, reasoning style, behavior |
| **RAG** | *What* the model knows — injects external knowledge at inference time |
