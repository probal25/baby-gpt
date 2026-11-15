# Nano-GPT: Bigram Language Model Training

![PyTorch Logo](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project implements a **Transformer-based language model** from scratch using PyTorch. Inspired by the original [GPT](https://paperswithcode.com/method/gpt), this code trains a simple transformer architecture on text data to learn language patterns and generate coherent text.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start Guide](#quick-start-guide)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Component Breakdown](#component-breakdown)
- [Configuration Parameters](#configuration-parameters)
- [How Training Works](#how-training-works)
- [Text Generation Process](#text-generation-process)
- [Hyperparameter Choices Explained](#hyperparameter-choices-explained)
- [Getting Started Guide for Beginners](#getting-started-guide-for-beginners)
- [Common Issues & Tips](#common-issues--tips)

---

## 🎯 Overview

**What is this?**

A neural network trained to predict the next character in a sequence of text, using advanced **attention mechanisms**. It's similar to how humans learn language patterns - by seeing lots of examples and figuring out what comes next.

**Key Concepts:**
- **Transformer Architecture**: Uses "attention" to weigh important parts of previous text
- **Positional Embeddings**: Tells the network which position a character is in (like word order matters)
- **Tokenization**: Converting letters/words into numbers the network can understand
- **Training Loop**: Shows examples, learns from mistakes, improves over time

---

## 🚀 Quick Start Guide

### Prerequisites

```bash
# Check PyTorch installation
pip install torch

# Or use conda (recommended for macOS)
conda install pytorch-cu121 -c pytorch -c nvidia  # CUDA enabled
conda install pytorch-cpu -c pytorch  # CPU only
```

### Running the Code

```bash
python bigram.py
```

**Note:** This requires an `input.txt` file in the same directory. Replace it with your own text files for custom training!

---

## 🏗️ Architecture Deep Dive

### The Big Picture

Think of this as a **multi-story building**, where each story (layer) learns something different about language patterns:

```
┌─────────────────────────────────────────┐
│  Output Layer                            │
│     (Predicts next character)            │
└─────────────────────────────────────────┘
              ↑
              │
┌─────────────────────────────────────────┐
│  Normalization Layers                     │
│      (Keeps values stable)               │
└─────────────────────────────────────────┘
              ↑
              │
┌─────────────────────────────────────────┐
│  Attention Mechanism                      │
│     (6 separate "viewpoints")           │
└─────────────────────────────────────────┘
              ↑
              │
┌─────────────────────────────────────────┐
│  Feed Forward Network                     │
│     (Non-linear transformations)         │
└─────────────────────────────────────────┘
              ↑
              │
┌─────────────────────────────────────────┐
│  Base Layers                              │
│      (Embedding & Position Encoding)    │
└─────────────────────────────────────────┘
```

### Components Explained

#### 1. **Token Embedding** (`token_embedding_table`)
- Converts text characters to vectors (numerical representations)
- Each character gets a unique vector in an n-dimensional space
- Similar characters live closer together in this space

#### 2. **Positional Embedding** (`position_embedding_table`)
- Tells the network where each token is in the sequence
- Without this, the network wouldn't know that "dog bites man" ≠ "man bites dog"

#### 3. **Multi-Head Attention** (`blocks`)
- The heart of the transformer! Allows the network to **attend to important parts** of previous text
- Multiple "heads" (6 different viewpoints) look at data from different perspectives:
  - One head might focus on word relationships
  - Another focuses on patterns
  - Yet another might catch grammatical structures

#### 4. **Feed Forward Network** (`ffwd`)
- A small neural network that applies non-linear transformations
- Lets the model learn complex patterns beyond simple attention

#### 5. **Output Projection** (`lm_head`)
- Converts learned representations into probabilities for next characters
- Gives you a distribution of what might come next

---

## 🔧 Component Breakdown

| Component | Lines | Purpose | Analogy |
|-----------|-------|---------|---------|
| `Head` | 81-108 | Self-attention mechanism | Focusing attention on important details |
| `MultiHeadAttention` | 110-123 | Combines multiple attention heads like experts | Team of specialists working together |
| `FeedForward` | 125-137 | Non-linear transformation | Learning complex relationships |
| `Block` | 140-154 | Single transformer layer | Processing unit combining all mechanisms |
| `BigramlanguageModel` | 156-185 | Full model architecture | The entire building structure |

---

## ⚙️ Configuration Parameters

```python
batch_size = 64              # Number of sequences processed at once
block_size = 256             # Maximum context length (most recent 256 chars)
max_iters = 10000            # Total training iterations
eval_interval = 500          # Every 500 iterations, evaluate on validation data
learning_rate = 3e-3         # How much weights adjust per step (step size)
n_embed = 384                # Embedding dimension (size of each vector)
n_head = 6                   # Number of attention heads in parallel
n_layer = 6                  # Depth of the model (number of transformer layers)
dropout = 0.2                # Probability of randomly dropping tokens (prevents overfitting)
eval_iters = 200             # Samples used for evaluation each step
```

---

## 🔄 How Training Works

### Step-by-Step Process

1. **Data Preparation**
   - Read text from `input.txt`
   - Create character-to-integer mapping (e.g., "a"→0, "b"→1, etc.)
   - Split data 90% training / 10% validation

2. **Creating Training Pairs**
   ```
   Example: "the cat sat on the mat"
   
   Input (x):    [t, h, e, , c, a, t]  # Last 8 characters
   Target (y):   [h, e, , c, a, t, , ]  # Next position's character
   ```

3. **Forward Pass**
   - Input goes through all layers: embedding → blocks → output
   - Model predicts probability distribution for next token

4. **Loss Calculation**
   - Compare predictions to actual targets
   - Cross-entropy measures how "surprised" the model was

5. **Backpropagation**
   - Calculate gradients (how much each weight contributed to error)
   - Update weights in direction that minimizes loss

6. **Iterate**
   - Repeat steps 3-5 thousands of times

7. **Evaluation**
   - Test on validation set every 500 iterations
   - Monitor training/val loss to check for overfitting

### Visualizing the Loss Curve

```
Training Loss: ████████████████████ (decreases over time)
Validation Loss: ███████████░░░░░░░ (stays similar, may diverge)

Good signs:
- Both losses decrease together
- Gap between train/val stays small

Bad signs:
- Validation loss rises after training loss falls (overfitting!)
```

---

## ✨ Text Generation Process

### The Generative Loop (`generate` method)

```python
def generate(self, idx, max_new_tokens):
    # For each step:
    1. Feed last 256 tokens into model
    2. Get probability distribution over all characters
    3. Sample next character from this distribution
    4. Append and repeat up to max_new_tokens times
    
# Returns complete generated text!
```

**Key Concept:** Unlike deterministic models, it uses **probabilistic sampling**. Every time you generate "the cat sat" → "on", the next word might vary slightly due to randomness.

### Generation Examples

Given context: `"Once upon a"` → Possible outputs:
- `"time there was a prince"`
- `"night and a dragon lived nearby"`
- `"tale of a brave king named Gerald"`

**Note:** It generates text on-the-fly from the input.txt, not from pre-trained knowledge!

---

## 📊 Hyperparameter Choices Explained

| Parameter | Value | Why This Choice? |
|-----------|-------|------------------|
| `n_embed` (embedding size) | 384 | Balances model capacity vs. memory usage |
| `n_head` (heads) | 6 | More heads = better context understanding, but uses more memory |
| `n_layer` (depth) | 6 | Deep enough to learn complex patterns, shallow enough for quick training |
| `learning_rate` | 0.003 | Standard AdamW learning rate for this type of model |
| `dropout` | 0.2 | Common value; prevents memorizing the training data |

**Trade-offs:**
- Larger = more powerful but slower to train, uses more memory
- Smaller = faster but may underfit complex patterns

---

## 📚 Getting Started Guide for Beginners

### Why Should You Understand This?

Think of it like **learning a second language**:
1. You read lots of examples (text)
2. Your brain figures out patterns (training)
3. Eventually you can generate new text (inference)

This code shows **how that actually works** under the hood!

### Key Concepts to Know

#### 1. What is an "embedding"?
- Converting letters/words to numbers using vectors
- Example: "dog" = [0.8, -0.2, 0.5...] → vector representation
- Similar words have similar vectors (closer in mathematical space)

#### 2. What is "attention"?
- Like reading a book and focusing on important parts
- Each token can attend to ANY other token in the sequence
- Helps understand context ("cat" could be an animal or pet!)

#### 3. What does "dropout" do?
- Randomly sets some neurons offline during training
- Prevents the model from over-relying on specific features
- Like having a team where everyone can contribute different solutions

### Quick Experiment Ideas

```python
# Modify vocabulary size
vocab_size = len(chars)  # Use your custom chars

# Change context length
block_size = 128  # Use shorter sequences initially
```

---

## ⚠️ Common Issues & Tips

### Issue: Out of Vocabulary Characters

**Symptom:** Network crashes when encountering characters it hasn't seen.

**Fix:** Either expand `chars` list or handle unknown tokens gracefully.

### Issue: Very Long Sequences Truncated

**Cause:** Using `block_size=256` limits how far back the model can "look".

**Solution:** Increase context length (requires more compute) for longer sequences.

### Issue: Memory Issues

**Symptom:** Running out of memory on CPU.

**Tip:** Reduce `n_embed`, `n_layer`, or use smaller batch size.

---

## 📖 For Different Readers

### For Non-Technical Readers (Students, Curious Minds)

> **This code teaches computers to read like humans do.**
>
> - Think of the network as a student who's been reading books for years
> - The "training" is just showing them examples and letting them learn patterns
> - Over time, they figure out grammar, common word choices, and context
> - When you give it some words, it generates new text that feels natural
>
> **Why does this matter?** These ideas power everything from chatbots to translating speech to making art. Understanding the basics helps appreciate how AI works under the hood!

### For Technical Readers (Researchers, Engineers)

This implementation provides:

- **Pure PyTorch transformer architecture** with attention mechanisms
- **Custom data loading pipeline** for text preprocessing
- **Standard GPT-style training loop** with early stopping via validation loss
- **Probabilistic generation** using softmax sampling

**Key architectural choices:**
- Multi-head self-attention with causal masking (positional triangular mask)
- Residual connections around attention and FFN layers
- Layer normalization for stable gradients
- Feed-forward network with ReLU activation

For further reading, see the original paper: [GPT-2: A New Approach to Large-Scale Language Generation](https://arxiv.org/abs/2005.14185)

---

## 🚧 Future Enhancements

Possible improvements for this architecture:
- Add positional encodings with sinusoids (like GPT-2)
- Use RoPE (Rotary Positional Embeddings)
- Implement FlashAttention for faster inference
- Add beam search for better generation quality
- Enable parallel evaluation across multiple GPUs

---

## 📝 License

MIT License - feel free to use this for learning and experimentation!
