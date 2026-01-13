#!/bin/bash

echo "PROJECT ALEPH BUILD SCRIPT"
echo "=========================="

# Create output directory
mkdir -p bin

# Check for compiler
if command -v g++ &> /dev/null; then
    COMPILER="g++"
elif command -v clang++ &> /dev/null; then
    COMPILER="clang++"
else
    echo "Error: GCC or Clang not found."
    exit 1
fi

echo "Using compiler: $COMPILER"

# Flags for AVX-512 and OpenMP
# Note: For MacOS users with Clang, OpenMP might require -Xpreprocessor -fopenmp and -lomp
FLAGS="-O3 -mavx512f -mavx512bw -fopenmp"

# Compile
echo "Compiling src/aleph_kernel.cpp..."
$COMPILER $FLAGS src/aleph_kernel.cpp -o bin/aleph_engine

if [ $? -eq 0 ]; then
    echo "Build Successful."
    echo "Run with: ./bin/aleph_engine"
else
    echo "Build Failed."
fi