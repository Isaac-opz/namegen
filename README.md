# namegen

namegen is a Rust CLI for training and generating realistic random names using a compact GPT-style model and deterministic sampling controls.

## Documentation

- Getting started: [getting-started.md](getting-started.md)
- Architecture: [architecture.md](architecture.md)

## Highlights

- Clear CLI with train and gen commands
- Reusable library modules behind src/lib.rs
- Checkpoint persistence with serde and bincode
- Deterministic generation and training support through explicit seed inputs
- Temperature, top-k, and top-p sampling controls
- Quality filtering for generated outputs
- Optional Config.toml defaults

## Install

```bash
cargo install --path .
```

## Quick Start

Train:

```bash
namegen train \
  --dataset input.txt \
  --epochs 2 \
  --learning-rate 0.01 \
  --save-model assets/my-model.bin \
  --seed 42
```

Generate:

```bash
namegen gen \
  --count 20 \
  --temperature 0.8 \
  --top-k 20 \
  --top-p 0.9 \
  --seed 42 \
  --load-model assets/my-model.bin
```

If load-model is not provided, namegen uses assets/pretrained.bin.

## Config Defaults

Config.toml is optional. Values in defaults are used unless overridden by CLI flags.

```toml
[defaults]
epochs = 2
learning_rate = 0.01
temperature = 0.8
count = 20
top_k = 20
top_p = 0.9
```

## Development Checks

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Logging

```bash
RUST_LOG=info namegen train --dataset input.txt --save-model assets/tmp.bin
RUST_LOG=debug namegen gen --load-model assets/pretrained.bin --seed 42
```
