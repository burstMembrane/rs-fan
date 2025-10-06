#!/bin/bash
# Build with maximum optimizations for AMD Threadripper 3970X

export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx,+avx2,+fma,+sse4.2 -C llvm-args=-enable-machine-outliner=never"

echo "Building with optimizations:"
echo "  - Target CPU: native (Threadripper 3970X)"
echo "  - SIMD: AVX2, FMA, SSE4.2"
echo "  - LTO: thin"
echo "  - Allocator: jemalloc (binary only)"
echo "  - Vectorized f32->i16 conversion"
echo ""

# Build binary
cargo build --release --bin resamplefan

# Build Python module
echo ""
echo "Building Python module..."
maturin develop --release

echo ""
echo "Build complete!"
echo "  Binary: ./target/release/resamplefan"
echo "  Python module installed in venv"
