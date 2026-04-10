# Getting Started

## Who this is for

This guide is for contributors and users who want to build, test, train, and generate names with namegen.

## Prerequisites

- Rust toolchain installed through rustup
- Cargo available in your shell
- macOS, Linux, or Windows with a standard terminal

Optional but recommended:

- A release build for faster training and generation
- RUST_LOG configured during troubleshooting

## Quick Setup

1. Clone the repository.
2. Change into the project directory.
3. Build the project.

```bash
cargo build
```

4. Run tests.

```bash
cargo test
```

## Install the CLI locally

```bash
cargo install --path .
```

This installs the namegen binary into your Cargo bin directory.

## Train a model

Use the included dataset or provide your own one-name-per-line file.

```bash
namegen train \
  --dataset input.txt \
  --epochs 2 \
  --learning-rate 0.01 \
  --save-model assets/my-model.bin \
  --seed 42
```

Notes:

- Increasing epochs generally improves quality but increases runtime.
- Keep seed fixed while tuning to compare changes fairly.

## Generate names

```bash
namegen gen \
  --count 20 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9 \
  --seed 42 \
  --load-model assets/my-model.bin
```

If load-model is omitted, the default is assets/pretrained.bin.

## Use Config.toml defaults

Create or edit Config.toml in project root.

```toml
[defaults]
epochs = 2
learning_rate = 0.01
temperature = 0.8
count = 20
top_k = 20
top_p = 0.9
```

CLI flags override Config.toml values.

## Development Workflow

Recommended local checks before opening a PR:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo run -- --help
```

For deterministic verification, run generation twice with the same seed and compare outputs.

## Common Tasks

## Run with logs

```bash
RUST_LOG=info namegen train --dataset input.txt --save-model assets/tmp.bin
RUST_LOG=debug namegen gen --load-model assets/pretrained.bin --seed 42
```

## Use a custom dataset

- Keep one sample per line.
- Remove empty lines.
- Prefer alphabetic names for best quality filtering results.

## Troubleshooting

- If command not found after install, ensure Cargo bin path is on PATH.
- If generation returns no names, relax quality settings or sample more candidates.
- If checkpoint load fails, retrain and save with the current version.

## Next Reading

- See architecture.md for system design details.
- See README.md for the project overview and command references.
