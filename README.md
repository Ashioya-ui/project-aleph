PROJECT ALEPH // INTEGER RESONANCE ARCHITECTURE

CLASSIFICATION: PUBLIC RELEASE (PHASE 3)
ARCHITECT: William Alubokho Ashioya
TARGET: CONSUMER CPU (AVX-512)

"This is a physics-first AI architecture designed to break the NVidia monopoly by enabling massive resonant agents on consumer CPUs. The goal is not to simulate a brain on a GPU cluster, but to build a resonator that fits in a pocket."

1. Overview

Project Aleph represents a paradigm shift from Matrix Multiplication ($O(N^2)$) to Holographic Resonance ($O(N \log N)$).

By fusing the Fast Walsh-Hadamard Transform (FWHT) for mixing, the Discrete Hartley Transform (DHT) for phase/position, and Alman-Rao Non-Rigid Sparsity for memory, Aleph runs large-scale resonant agents entirely on CPU integers, bypassing the VRAM bottleneck of traditional Transformers.

2. Repository Contents

src/aleph_kernel.cpp (The Metal)

The production-grade C++ engine.

AVX-512 Optimized: Uses 512-bit registers to process 32 integers per clock cycle.

Integer Native: Replaces floating point math with high-speed integer addition/subtraction.

Sparse Adjacency: Implements "Engram" memory via direct pointer lookup ($O(K)$) rather than dense multiplication.

src/aleph_practical_engine.py (The Reference)

The PyTorch prototype for training and architectural verification.

Implements the AlephPracticalEngine class.

Simulates the "Hartley Twist" and "Sparse Gather" using advanced tensor indexing.

Used to train weights before exporting them to the C++ kernel.

docs/ALEPH_MANUAL.md

The official implementation guide, covering:

Bipolar initialization strategies.

Sparsity threshold tuning.

Deployment instructions.

3. Quick Start

Prerequisites

C++: GCC or Clang with OpenMP and AVX-512 support.

Python: PyTorch 2.0+ (for the reference engine).

Building the Production Kernel

Run the included build script:

./build.sh


Or compile manually:

g++ -O3 -mavx512f -mavx512bw -fopenmp src/aleph_kernel.cpp -o aleph_engine


Running

./aleph_engine


Expected Output:

PROJECT ALEPH :: PRODUCTION KERNEL (PATCHED)
>> AVX-512 DETECTED. OMP DETECTED.
>> INJECTING SIGNAL...
>> RESONANCE SUCCESS.


4. Performance Notes

Throughput: The kernel processes ~32 operations per cycle vs 1-8 for standard scalar code.

Latency: Data resides in L2 Cache (KB range) rather than VRAM (GB range).

Safety: Includes 64-bit accumulator patching to prevent integer overflow during resonance.

Project Aleph is a "Physics-First" approach to AI. We do not store data; we store interference patterns.
