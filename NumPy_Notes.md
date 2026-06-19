# NumPy Notes — Foundations for ML / Data Science

## 1. What is NumPy?

**NumPy = Numerical Python**

Provides:
- Fast arrays (`ndarray`)
- Vectorized operations
- Matrix operations & linear algebra
- The foundation for Pandas, Scikit-Learn, PyTorch, TensorFlow

```python
import numpy as np
```

---

## 2. Creating Arrays

**1D Array**
```python
arr = np.array([1, 2, 3, 4])
# [1 2 3 4]
```

**2D Array**
```python
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
# [[1 2 3]
#  [4 5 6]]
```

---

## 3. Shape — Most Important Attribute

```python
arr.shape
```

```python
arr = np.array([[1,2,3],[4,5,6]])
arr.shape   # (2, 3)  → 2 rows, 3 columns
```

**In ML terms:** shape = `(samples, features)`
> e.g. 1000 students × 5 features → `shape = (1000, 5)`

---

## 4. Dimensions, Size, Dtype

| Attribute | Meaning | Example | Output |
|---|---|---|---|
| `arr.ndim` | number of dimensions | `[[1,2],[3,4]]` | `2` |
| `arr.size` | total element count | `[[1,2,3],[4,5,6]]` | `6` |
| `arr.dtype` | data type | `[1,2,3]` | `int64` |

Convert type:
```python
arr.astype(float)
```

---

## 5. Special Arrays

```python
np.zeros((2,3))        # [[0. 0. 0.] [0. 0. 0.]]
np.ones((2,3))         # all ones, same shape
np.arange(0,10)         # [0 1 2 3 4 5 6 7 8 9]
np.linspace(0, 1, 5)    # [0.   0.25 0.5  0.75 1.]
```

| Function | Purpose |
|---|---|
| `zeros` | array filled with 0s |
| `ones` | array filled with 1s |
| `arange(start, stop)` | range of values (step-based) |
| `linspace(start, stop, n)` | `n` evenly spaced values |

---

## 6. Indexing

```python
arr = np.array([10,20,30,40])
arr[0]          # 10
```

2D:
```python
arr = np.array([[1,2,3],[4,5,6]])
arr[1][2]       # 6
arr[1,2]        # 6  ← preferred, cleaner syntax
```

---

## 7. Slicing

```python
arr = np.array([10,20,30,40,50])
arr[1:4]        # [20 30 40]
```

2D — all rows, one column:
```python
arr[:, 1]       # column index 1 → [2 5]
```

---

## 8. Reshape (Very Important)

```python
arr = np.arange(6)
arr.reshape(2,3)
# [[0 1 2]
#  [3 4 5]]
```

**Rule:** total elements must stay the same.
- ✅ `6 → (2,3)` or `6 → (3,2)`
- ❌ `6 → (2,4)` — invalid, element count mismatch

---

## 9. Flatten

Converts multi-dimensional → 1D.

```python
arr.flatten()
# [[1,2],[3,4]]  →  [1 2 3 4]
```

---

## 10. Vectorized Operations (NumPy's Superpower)

No loops needed:

```python
arr = np.array([1,2,3])
arr + 10        # [11 12 13]
arr * 2         # [2 4 6]
```

**Element-wise array ops:**
```python
a = np.array([1,2,3])
b = np.array([4,5,6])

a + b   # [5 7 9]
a * b   # [4 10 18]  ← element-wise, NOT matrix multiplication
```

---

## 11. Aggregations

| Function | Purpose |
|---|---|
| `arr.sum()` | total |
| `arr.mean()` | average |
| `arr.max()` | maximum |
| `arr.min()` | minimum |
| `arr.std()` | standard deviation |

Heavily used in ML preprocessing (normalization, scaling, stats).

---

## 12. Axis — The Concept That Confuses Everyone

Visualize a matrix:
```
      Columns
        ↓
      0  1  2
    +---------
0 → | 1  2  3
1 → | 4  5  6
Rows
```

### `axis=0` → move **DOWN** the rows (vertical ↓)
```python
arr.sum(axis=0)
```
```
1 + 4 = 5
2 + 5 = 7
3 + 6 = 9
→ [5 7 9]
```
**Meaning:** collapse rows, keep columns.

### `axis=1` → move **ACROSS** the columns (horizontal →)
```python
arr.sum(axis=1)
```
```
1+2+3 = 6
4+5+6 = 15
→ [6 15]
```
**Meaning:** collapse columns, keep rows.

### Why "axis"?
For `shape = (2, 3)`:
- Axis 0 = the 2 rows
- Axis 1 = the 3 columns

### Quick memory trick
```
axis=0 → Vertical ↓   → operate on rows, keep columns
axis=1 → Horizontal → → operate on columns, keep rows
```

This rule applies identically to `sum()`, `mean()`, `max()`, `min()`, `std()`, etc.

### ML Example
```python
X = np.array([
    [10, 2],
    [20, 4],
    [30, 6]
])   # shape (3,2) → 3 samples, 2 features

X.mean(axis=0)
# Feature1: (10+20+30)/3 = 20
# Feature2: (2+4+6)/3   = 4
# → [20. 4.]
```
This is exactly how you compute **per-feature mean** in ML preprocessing.

---

## 13. Boolean Masking

Very common in data filtering.

```python
arr = np.array([10,20,30,40])
arr > 25            # [False False True True]
arr[arr > 25]       # [30 40]
```

---

## 14. Broadcasting — Most-Asked Concept

NumPy automatically expands dimensions to make shapes compatible.

```python
arr = np.array([[1,2,3],[4,5,6]])
arr + 10
# [[11 12 13]
#  [14 15 16]]
```

**Matrix + vector:**
```python
[[1,2,3],
 [4,5,6]]
+
[10,20,30]
=
[[11,22,33],
 [14,25,36]]
```
The smaller array is "stretched" to match the larger one's shape — this automatic expansion is **broadcasting**.

---

## 15. Random Numbers

```python
np.random.rand(3)              # [0.34 0.72 0.11]  → uniform floats [0,1)
np.random.randint(1, 10, size=5)  # [2 5 7 1 9]      → random integers
```

---

## 16. Matrix Multiplication

For real matrix multiplication (not element-wise), use `@` or `np.dot()`:

```python
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

A @ B
# [[19 22]
#  [43 50]]
```

---

## 17. The Standard ML Preprocessing Workflow

```python
import numpy as np

X = np.array([
    [10, 2],
    [20, 3],
    [30, 5]
])

print(X.shape)               # check shape

mean = X.mean(axis=0)        # per-feature mean
print(mean)

normalized = X - mean        # center the data (broadcasting in action)
print(normalized)
```

**General pipeline:**
```
Load data → Convert to NumPy array → Check shape → Reshape
   → Normalize → Train model
```

---

## 18. Priority Learning Order

```
1. array()
2. shape
3. ndim
4. reshape()
5. Indexing & slicing
6. Vectorized operations
7. Broadcasting
8. sum(), mean(), max()
9. Boolean masking
10. Matrix multiplication (@)
```

Master these 10 and you can read almost any NumPy code in ML, data science, or the internals of PyTorch/TensorFlow.

---

## 19. One-Line Takeaways

- **Shape** tells you `(samples, features)` — always check it first.
- **axis=0 ↓ rows, axis=1 → columns** — this single rule unlocks `sum`, `mean`, `max`, `min`, `std`.
- **Vectorization** replaces loops — operations apply to the whole array at once.
- **Broadcasting** lets NumPy combine arrays of different shapes without manual loops.
- `*` = element-wise multiply, `@` = real matrix multiplication — don't confuse them.
