PROJECT ALEPH: IMPLEMENTATION MANUAL

CLASSIFICATION: ENGINEERING GUIDELINE // PHASE 3
VERSION: 1.0.0
TARGET HARDWARE: CONSUMER CPU (AVX-512 SUPPORT RECOMMENDED)

1. Executive Summary

Project Aleph is a Linear-Time, Integer-Preferred Architecture designed to run Large Language Models on limited hardware. It replaces the standard Matrix Multiplication ($O(N^2)$) with three fused technologies:

Fast Walsh-Hadamard Transform (FWHT): For mixing information ($O(N \log N)$).

Discrete Hartley Transform (DHT): For preserving sequence order (Phase).

Alman-Rao Sparse Decomposition: For memory storage ($O(N \cdot K)$).

2. Architecture Integration

The AlephPracticalEngine is a Layer Replacement. You use it to replace nn.Linear or nn.SelfAttention blocks.

Constraints

Power of 2 Dimensions: The Fast Walsh-Hadamard logic relies on recursive splitting. Your embedding dimension (d_model) MUST be a power of 2 (e.g., 512, 1024).

Bipolar Input Preference: The engine works best when inputs are normalized to roughly $\{-1, 1\}$. Use nn.LayerNorm before passing data into Aleph.

3. The Data Pipeline

Step 1: Embedding & Bipolarization

Instead of initializing random floats, initialize random integers $\{-1, 1\}$.

# Recommended Embedding Initialization
embedding = nn.Embedding(vocab_size, 512)
with torch.no_grad():
    embedding.weight.data.sign_() 


Step 2: The Forward Pass (Wave Dynamics)

When you pass data through the engine, three physical processes occur:

Scattering (Fusion): The fast_hadamard_hartley_fusion function spreads the input token's info across the entire vector.

Filtering (Alman-Rao): The engine applies the "Non-Rigid" filter.

Spectral Weights: These handle grammar/syntax (Global rules).

Engram Indices: These look up specific memories (Sparse facts).

Gathering: The wave is collapsed back to the time domain.

4. Tuning The "Engram" (Sparsity)

The sparsity_threshold is your main knob for performance vs. intelligence.

Threshold

Active Params (1024 dim)

Behavior

Hardware

0.01 (1%)

~10 per neuron

Generalist. Good at grammar, weak at facts.

Raspberry Pi / Old Laptop

0.05 (5%)

~50 per neuron

Balanced. Similar to Llama-7B density.

Standard Laptop

0.10 (10%)

~100 per neuron

Specialist. High recall.

High-End Workstation

5. Troubleshooting

Issue: RuntimeError regarding tensor sizes.

Cause: Your dimension is not a power of 2.

Fix: Resize your inputs or use F.pad.

Issue: "Phase information lost"

Cause: dim is too small (e.g., 2 or 4) for the Hartley rotation to create distinct interference patterns.

Fix: Use dim >= 128.

Issue: Output values are clamped (C++ Kernel).

Cause: The model has drifted too far from unity.

Fix: Ensure your Python training uses weight_decay to keep engram_values small.