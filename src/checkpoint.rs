use crate::model::{GPT, ModelConfig};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checkpoint {
    version: u32,
    config: ModelConfig,
    tokenizer: Tokenizer,
    state_dict: HashMap<String, Vec<Vec<f64>>>,
}

pub fn save_checkpoint(path: &Path, gpt: &GPT, tokenizer: &Tokenizer) -> Result<()> {
    let mut state_dict = HashMap::new();
    for (name, matrix) in &gpt.state_dict {
        let rows: Vec<Vec<f64>> = matrix
            .iter()
            .map(|row| row.iter().map(|&v| gpt.graph.node_data(v)).collect())
            .collect();
        state_dict.insert(name.clone(), rows);
    }

    let ckpt = Checkpoint {
        version: 1,
        config: gpt.cfg,
        tokenizer: tokenizer.clone(),
        state_dict,
    };

    let bytes = bincode::serde::encode_to_vec(&ckpt, bincode::config::standard())
        .context("failed to serialize checkpoint")?;
    std::fs::write(path, bytes)
        .with_context(|| format!("failed to write checkpoint to {}", path.display()))?;
    Ok(())
}

pub fn load_checkpoint(path: &Path) -> Result<(GPT, Tokenizer)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read checkpoint {}", path.display()))?;
    let (mut ckpt, _): (Checkpoint, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .context("failed to deserialize checkpoint")?;

    let gpt = GPT::from_weights(ckpt.tokenizer.vocab_size(), ckpt.config, ckpt.state_dict)
        .context("failed to rebuild model from checkpoint")?;
    ckpt.tokenizer.rebuild_index();
    Ok((gpt, ckpt.tokenizer))
}
