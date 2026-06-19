# PyTorch Notes — From NumPy to Deep Learning

## 1. The Big Picture

> **PyTorch = NumPy + Automatic Differentiation + GPU Support + Deep Learning Tools**

Used for: Neural Networks, Deep Learning, Computer Vision, NLP/LLMs, Reinforcement Learning, Generative AI.

Most AI research and production systems run on PyTorch. Libraries like Hugging Face Transformers, LangChain, and LlamaIndex are PyTorch-based under the hood.

---

## 2. Why Not Just NumPy?

Training any model needs this loop:

```
Compute output → Calculate error → Compute gradients → Update weights → Repeat millions of times
```

- **NumPy**: you must derive and code the gradient formulas yourself.
- **PyTorch**: gradients are computed automatically via **Autograd**.

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)   # 4   (dy/dx = 2x = 2*2)
```

PyTorch did the calculus — no manual derivatives needed.

---

## 3. Worked Example: Predicting House Prices

### Step 1 — Data (NumPy)
```python
import numpy as np
X = np.array([1000, 1200, 1500])   # area (features)
y = np.array([50, 60, 75])         # price (target)
```

### Step 2 — Simple model: `Price = Area × w`
```python
w = 0.01
pred = X * w        # [10, 12, 15]   ← way off from [50, 60, 75]
```

### Step 3 — Measure error
```python
error = pred - y          # [-40, -48, -60]
loss = np.mean(error**2)  # Mean Squared Error
```

### Step 4 — The problem
How much should `w` change? `0.02`? `0.1`? `1`? We need direction **and** magnitude → that's a **gradient**.

```
d(loss)/d(w)  →  tells us which way and how much to move w
```

- **In NumPy**: you derive this formula by hand. Fine for one equation — impossible for a model with billions of parameters (like GPT-4).
- **In PyTorch**: just set `requires_grad=True` and call `.backward()`.

---

## 4. Core Concept #1 — Tensor

A **Tensor** = generalized array = NumPy array + gradient tracking + GPU support.

| Dimensions | Example | Used for |
|---|---|---|
| 0D | scalar | single number |
| 1D | vector | list of features |
| 2D | matrix | tabular data |
| 3D | tensor | single image (H×W×C) |
| 4D | tensor | batch of images |

```python
tensor = torch.tensor([1, 2, 3])   # same idea as np.array([1,2,3])
```

**Mental model:**
- NumPy array = stores numbers
- PyTorch tensor = stores numbers **+** computation graph **+** gradients **+** can run on GPU

> NumPy = Calculator. PyTorch Tensor = Calculator + Learning Ability.

---

## 5. Core Concept #2 — GPU Acceleration

| | NumPy | PyTorch |
|---|---|---|
| Hardware | CPU only | CPU + GPU |

```python
device = "cuda"
tensor = tensor.to(device)   # now runs on GPU
```

Example impact: a job taking **10 hours on CPU** might take **30 minutes on GPU** (hardware-dependent).

---

## 6. Core Concept #3 — Autograd (Automatic Differentiation)

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)   # 4
```

PyTorch silently builds a **computation graph**:

```
x ───► x² ───► y
```

For something more complex:

```python
y = ((x*3) + 5)**2
```

Graph:
```
x → ×3 → +5 → square → y
```

Calling `y.backward()` automatically applies the **chain rule** across every step of the graph to compute gradients. This is the foundation of deep learning — no manual calculus, even for models with millions/billions of parameters.

---

## 7. Core Concept #4 — Neural Network Modules (`torch.nn`)

Instead of writing the math from scratch:

```python
import torch.nn as nn
model = nn.Linear(10, 1)   # creates y = Wx + b automatically
```

PyTorch auto-creates and manages the weight `W` and bias `b` for you.

---

## 8. Core Concept #5 — Loss Function

Measures how wrong the prediction is.

```python
criterion = nn.MSELoss()
loss = criterion(predictions, targets)
```

Example: prediction = 55, actual = 60 → loss reflects that gap (model is wrong, needs to improve).

---

## 9. Core Concept #6 — Optimizer

Updates the weights using the gradient.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Intuition:**
- Gradient → tells the *direction*
- Optimizer → takes the *step*

```
old weight (0.5) + gradient (-0.1) → new weight (0.51)
```

---

## 10. The Training Loop (memorize this — it's everywhere)

```python
for epoch in range(epochs):
    outputs = model(X)              # forward pass / prediction
    loss = criterion(outputs, y)    # measure error
    optimizer.zero_grad()           # clear old gradients
    loss.backward()                 # compute new gradients (autograd)
    optimizer.step()                # update weights
```

### Full cycle visualized
```
Input Data → Tensor → Model → Prediction → Loss → backward()
   → Gradients → Optimizer → Updated Weights → Repeat thousands of times
```

This entire repeated process = **Gradient Descent / Training**.

---

## 11. Key PyTorch Modules Cheat Sheet

| Module | Purpose | Examples |
|---|---|---|
| `torch` | Tensor ops | `tensor()`, `zeros()`, `ones()`, `rand()` |
| `torch.nn` | NN layers | `Linear`, `Conv2d`, `LSTM`, `Transformer` |
| `torch.optim` | Optimizers | `Adam`, `SGD`, `AdamW` |
| `torch.utils.data` | Data loading | `Dataset`, `DataLoader` |

---

## 12. Comparison Tables

**PyTorch vs NumPy**

| Feature | NumPy | PyTorch |
|---|---|---|
| Arrays / Matrix math | ✅ | ✅ |
| GPU support | ❌ | ✅ |
| Gradients (autograd) | ❌ | ✅ |
| Neural networks | ❌ | ✅ |

> NumPy = Mathematics. PyTorch = Mathematics + Learning.