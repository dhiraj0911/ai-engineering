# Embeddings

## What Are Embeddings?

Embeddings are **numerical representations of data**.

They solve the core problem of how to convert unstructured data (text, image, audio) into numbers **in a way that preserves meaning**:

> Texts that mean similar things land near each other. Texts that mean different things land far apart.

Embeddings convert tokens into **dense, low-dimensional vectors** that capture semantic meaning.

---

## Cosine Similarity & Cosine Distance

**Cosine Similarity** is a metric that measures the cosine of the angle between two vectors projected in multi-dimensional space.

- The **smaller the angle** between two vectors, the **more similar** they are.

| Value | Meaning |
|-------|---------|
| `-1`  | Strongly opposite |
| `0`   | Orthogonal (unrelated) |
| `1`   | Highly similar |

**Cosine Distance** is the inverse:

```
Cosine Distance = 1 - Cosine Similarity
```

---

## Vector Databases

A **Vector Database** indexes and stores vector embeddings for fast retrieval and similarity search, with full CRUD capability.

### Relational DB vs. Vector DB

| | Relational DB | Vector DB |
|---|---|---|
| Query type | `find rows WHERE column = value` | `find the N vectors most similar to this query vector` |
| Use case | Exact structured lookup | Semantic / approximate nearest-neighbor search |

### How It Works — End-to-End Flow

```
Vectors => Indexing => Vector Database => Querying => Post-processing
```

**Step-by-step:**

1. Use an **embedding model** to create vector embeddings of the content you want to index.
2. Insert the vector embedding into the vector database, along with a **reference to the original content**.
3. At query time, use the **same embedding model** to embed the query, then search the vector database for similar embeddings — which map back to the original content.

> **Post-processing:** In some cases, the vector database retrieves a broader candidate set and applies a final re-ranking step to return the true nearest neighbors.
