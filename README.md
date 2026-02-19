# Custom SLM Pretraining with RoPE on TinyStories

A from-scratch implementation of a Small Language Model (SLM) with Rotary Positional Embeddings (RoPE) trained on the TinyStories dataset. This project serves as a hands-on exploration of core transformer mechanics, data pipeline management, and rigorous evaluation practices.

This is a learning project focusing on architectural understanding rather than SOTA performance. The model demonstrates potential for constrained domains but is not production-ready.

---

## Project Overview

This repository contains:
- **Custom transformer architecture** with RoPE positional encoding implementation
- **Complete data pipeline** including filtering, preprocessing, and efficient dataset packing  
- **Rigorous evaluation harness** with training dynamics analysis, attention visualization, and generation quality assessment
- **Full training loop** with early stopping and model checkpointing

---

## 1. Training Dynamics: Loss and Perplexity Evaluation

### Convergence Behavior

The model's training lifecycle was monitored across four key metrics:
- **Training Loss**: [View on W&B](https://wandb.ai/xenflashs-jkf-/small-llm-rope-tinystories/runs/cj7cm25f?nw=nwuserxenflashs&panelDisplayName=train_loss&panelSectionName=Charts)
- **Training Perplexity**: [View on W&B](https://wandb.ai/xenflashs-jkf-/small-llm-rope-tinystories/runs/cj7cm25f?nw=nwuserxenflashs&panelDisplayName=train_perplexity&panelSectionName=Charts)
- **Validation Loss**: [View on W&B](https://wandb.ai/xenflashs-jkf-/small-llm-rope-tinystories/runs/cj7cm25f?nw=nwuserxenflashs&panelDisplayName=val_loss&panelSectionName=Charts)
- **Validation Perplexity**: [View on W&B](https://wandb.ai/xenflashs-jkf-/small-llm-rope-tinystories/runs/cj7cm25f?nw=nwuserxenflashs&panelDisplayName=val_perplexity&panelSectionName=Charts)
- **Training Steps**: [View on W&B](https://wandb.ai/xenflashs-jkf-/small-llm-rope-tinystories/runs/cj7cm25f?nw=nwuserxenflashs&panelDisplayName=step&panelSectionName=Charts)
- **Epoch Progress**: [View on W&B](https://wandb.ai/xenflashs-jkf-/small-llm-rope-tinystories/runs/cj7cm25f?nw=nwuserxenflashs&panelDisplayName=epoch&panelSectionName=Charts)

### Key Findings

**Initial Descent**: The training loss exhibits a rapid descent in the early steps, dropping steeply from random initialization. This is characteristic of the TinyStories dataset; due to its restricted vocabulary (~1,500 words tailored for 3-4 year old comprehension), the model rapidly maps fundamental syntax and token distributions.

**Final Loss Values**: Training loss stabilized at approximately **1.47**, with validation loss converging to **~1.60**.

**Generalization**: The remarkably tight gap between training and validation loss curves indicates successful generalization. The validation curve closely tracks the training curve without diverging upward, demonstrating the model learned underlying grammatical rules rather than memorizing the training data.

### Perplexity Analysis

Perplexity measures the model's uncertainty when predicting the next token:

$$\text{PPL}(X) = \exp\left(-\frac{1}{T}\sum_{i=1}^{T} \log p_\theta(x_i \mid x_{\lt i})\right)$$

where $T$ is sequence length, $p_\theta(x_i | x_{<i})$ is the probability of token $x_i$ given preceding tokens.

**Validation Perplexity**: Dropped rapidly and plateaued at approximately **5.03**. This single-digit perplexity indicates high confidence in next-token predictions. While partially an artifact of the dataset's low vocabulary variance, it confirms the architecture successfully captures the target distribution.

---

## 2. Attention Mechanics and the Impact of RoPE

### Visualization References

- **Layer 5 Attention (Example 1)**: [Tom found a ball visualization](https://wandb.ai/xenflashs-jkf-/small-llm-rope-tinystories/runs/cj7cm25f/panel/7rqhwrcg2?nw=nwuserxenflashs)
- **Layer 5 Attention (Example 2)**: [Dragon story visualization](https://wandb.ai/xenflashs-jkf-/small-llm-rope-tinystories/runs/cj7cm25f/panel/8j24ef8ix?nw=nwuserxenflashs)

### Rotary Positional Embeddings (RoPE)

Unlike absolute positional encodings that add position-dependent vectors, RoPE encodes positional information through rotation matrices applied to query and key representations.

**Core Mechanism**: For a query at position $m$ and key at position $n$, the attention score depends solely on their relative distance $m - n$:

$$\text{RoPE}(x, m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}^T x$$

where $\theta$ varies across dimensions to encode position information across the embedding space.

### Observed Attention Patterns

**Local Context Preservation**: A strong local diagonal band is visible in Layer 5 attention maps, indicating RoPE successfully forces the model to heavily weight immediately preceding tokens—crucial for maintaining local syntax (e.g., adjectives correctly modifying adjacent nouns).

**Entity Tracking (Attention Sinks)**: In deeper layers, vertical bands of attention form around specific subject tokens (e.g., "Tom", "dragon"). The model utilizes these as "attention sinks" to maintain narrative consistency over the generation window, preserving entity references during generation.

---

## 3. Generation Quality Analysis

### Sample Generations

[View sample outputs on W&B](https://wandb.ai/xenflashs-jkf-/small-llm-rope-tinystories/runs/cj7cm25f/panel/g6o3f8r22?nw=nwuserxenflashs)

### Example 1: The "Timmy" Story

> "Timmy's friend accidentally hit Timmy's little sister and it made a cut on the front of his own face... 'Mom, can you help me fix the little sister?' Timmy asked... Let's go get some glue to fix your dad's boo-boo."

**Syntactic Strengths**: Grammar is near-flawless. The model successfully introduces conflict, seeks resolution, and formats dialogue with correct punctuation and narrative flow.

**Semantic Limitations**: Logical reasoning breaks down beyond syntax. The friend hits the sister but gets a cut on his own face. They use "glue" to fix a "sister", which morphs into "dad's boo-boo". The model understands narrative structure but lacks grounded, world-aware reasoning.

### Example 2: The "Forest" Story

> "She knew the dog was important. She knew that it was reliable. She ran home and told her mom. She was happy that she learned something important."

**Prompt Adherence**: Excellent thematic consistency with the prompt.

**Syntactic Pattern**: Heavy reliance on repetitive Subject-Verb-Object structures—a direct reflection of TinyStories' simplistic sentence construction patterns.

---

## 4. Architectural Insights & Future Optimizations

### Current Strengths

✓ Clean RoPE implementation with proven relative position encoding  
✓ Efficient data pipeline with proper tokenization and packing  
✓ Tight train/val loss gap indicating strong generalization  
✓ Interpretable attention patterns showing meaningful context capture  

### Recommended Improvements

**Depth vs. Width**: Consider increasing depth (more layers and heads) rather than width. TinyStories' restricted vocabulary (~1.5k words) without complex conceptual relationships means large embedding dimensions are underutilized. More layers would enable deeper non-linear combinations, yielding higher ROI for logical reasoning.

**Dataset Scaling**: Move beyond toddler-level semantics by scaling to larger, diverse corpora (FineWeb subsets, SlimPajama). This introduces richer vocabulary and complex grammatical relationships necessary for coherent reasoning.

**Grouped Query Attention (GQA)**: Reduce memory bandwidth for KV cache and accelerate inference by sharing key/value heads across multiple query heads. Particularly valuable for resource-constrained training.

**Extended Evaluation**:
- Downstream task finetuning (classification, summarization)
- Blue/ROUGE score analysis on structured text generation
- Probing tasks to measure syntactic vs. semantic understanding

---


## Model Configuration

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 50,257 (GPT-2) |
| Embedding Dimension | 640 |
| Attention Heads | 10 |
| Layers | 12 |
| Block Size (Context Length) | 512 tokens |
| Batch Size | 16 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Training Epochs | 10 |
| Evaluation Interval | 700 steps |
| Early Stopping Patience | 10 epochs |

---

## Key Takeaways

1. **RoPE is effective** for capturing relative position information in small models, evidenced by clean attention patterns and tight generalization gap.

2. **Architecture scales** well with the dataset's constraints—the model successfully learns TinyStories' restricted syntax despite simplistic vocabulary.

3. **Data quality matters**: The filtering pipeline (30-1024 token range, ≥20 words) creates a coherent training distribution that enables rapid convergence.

4. **Semantic understanding is hard**: Even with perfect syntax, the model lacks world knowledge. This highlights the importance of dataset diversity and scale for beyond-syntax reasoning.

---

## Citations & References

- Rotary Position Embeddings: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- TinyStories Dataset: [RonenEldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- Transformer Architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.10677)

---




