# Architecture

## Overview

namegen is a Rust CLI for training and generating realistic names with a compact GPT-style model and a custom scalar autograd graph.

The system is split into:

- CLI application in src/main.rs
- Reusable library API in src/lib.rs
- Domain modules for model, training, tokenization, sampling, quality filtering, and checkpoint I/O

## High-level Design

1. Input text lines are normalized into a document list.
2. Tokenizer builds a character vocabulary and BOS token.
3. GPT model is initialized or loaded from checkpoint.
4. Training performs forward and backward passes over token sequences.
5. Generation performs autoregressive sampling with temperature and optional top-k/top-p filtering.
6. Quality filtering removes low-quality or duplicate outputs.

## Module Map

- src/main.rs
  - CLI entrypoint using clap.
  - Implements train and gen commands.
  - Loads Config.toml defaults and initializes logging.

- src/lib.rs
  - Public module surface and re-exports.
  - Keeps binary code thin and testable.

- src/model.rs
  - Defines ModelConfig and GPT state.
  - Implements forward pass and key/value cache helpers.

- src/value.rs
  - Scalar graph engine and autodiff primitives.
  - Supports arena-style reuse through truncation and capacity controls.

- src/trainer.rs
  - Deterministic training loop with seeded RNG and Adam updates.
  - Produces TrainingSummary.

- src/tokenizer.rs
  - Character-level tokenizer.
  - Encodes and decodes tokens and maintains BOS token.

- src/sampling.rs
  - SamplingOptions validation.
  - Weighted sampling with optional top-k and top-p constraints.

- src/quality.rs
  - Candidate quality checks for length, alphabetic characters, and duplicates.

- src/checkpoint.rs
  - Checkpoint save/load using serde plus bincode.
  - Persists config, tokenizer, and model weights.

- src/config.rs
  - Optional Config.toml loader for defaults.

- src/errors.rs
  - Domain-specific error types for validation and model constraints.

## Runtime Flows

## Train flow

1. Read dataset file and build tokenizer.
2. Initialize GPT with seed-controlled RNG.
3. Iterate epochs and shuffled documents.
4. Compute average sequence loss and run backward pass.
5. Update parameters with Adam.
6. Save checkpoint artifact.

## Generate flow

1. Load checkpoint artifact.
2. Initialize seeded RNG for deterministic sampling.
3. Autoregressively produce next token logits.
4. Apply temperature and optional top-k and top-p.
5. Sample token, stop on BOS, build candidate name.
6. Run quality filter and print accepted names.

## Key Architectural Decisions

- Keep model logic in the library and keep CLI orchestration in main.
- Use a custom graph implementation to preserve educational transparency and direct control.
- Use explicit RNG seeds for reproducibility in training and generation.
- Persist full model state and tokenizer together for portable inference.
- Validate user inputs early and return contextual errors.

## Non-goals

- Large-scale distributed training.
- GPU acceleration.
- Tokenization beyond character-level modeling.

## Extension Points

- Add richer quality heuristics in src/quality.rs.
- Add alternative sampling policies in src/sampling.rs.
- Add new subcommands in src/main.rs while reusing src/lib.rs APIs.
- Add backward-compatible checkpoint schema evolution in src/checkpoint.rs.

## Reliability Practices

- No panic-driven runtime paths for expected errors.
- Seeded deterministic workflows for repeatable tests.
- Integration tests for tokenizer, sampling determinism, and checkpoint roundtrip.
- Context-rich errors at I/O and serialization boundaries.
