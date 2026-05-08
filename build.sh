#!/usr/bin/env bash
# Cross-compile ARM NEON Eisenstein benchmarks
# Requires: rustup target add aarch64-unknown-none
# For QEMU testing: sudo apt install qemu-user

set -euo pipefail

TARGET="aarch64-unknown-none"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "⚒️  ARM NEON Eisenstein Benchmark Builder"
echo "========================================="

# Check for cross-compilation toolchain
if ! rustup target list --installed | grep -q "$TARGET"; then
    echo "⚠️  Adding target $TARGET..."
    rustup target add "$TARGET"
fi

echo "📦 Building for $TARGET..."
cargo build --manifest-path "$REPO_DIR/Cargo.toml" \
    --target "$TARGET" \
    --release

echo "✅ Built: target/$TARGET/release/arm-neon-eisenstein-bench"

# Optional: run via QEMU if available
if command -v qemu-aarch64 &>/dev/null; then
    echo "🏃 Running via QEMU..."
    qemu-aarch64 "target/$TARGET/release/arm-neon-eisenstein-bench"
else
    echo "💡 To run: qemu-aarch64 target/$TARGET/release/arm-neon-eisenstein-bench"
fi

# Also build for native (x86_64 fallback — uses rdtsc instead of PMCCNTR)
echo ""
echo "📦 Building native fallback (x86_64, rdtsc-based)..."
cargo build --manifest-path "$REPO_DIR/Cargo.toml" --release
echo "🏃 Running native fallback..."
cargo run --manifest-path "$REPO_DIR/Cargo.toml" --release
