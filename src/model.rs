use crate::errors::NamegenError;
use crate::value::{Graph, Value};
use anyhow::{Context, Result};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const N_LAYER: usize = 1;
pub const N_EMBD: usize = 16;
pub const BLOCK_SIZE: usize = 16;
pub const N_HEAD: usize = 4;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub block_size: usize,
    pub n_head: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_layer: N_LAYER,
            n_embd: N_EMBD,
            block_size: BLOCK_SIZE,
            n_head: N_HEAD,
        }
    }
}

impl ModelConfig {
    pub fn head_dim(self) -> usize {
        self.n_embd / self.n_head.max(1)
    }
}

pub struct GPT {
    pub state_dict: HashMap<String, Vec<Vec<Value>>>,
    pub graph: Graph,
    pub cfg: ModelConfig,
}

impl GPT {
    pub fn new(vocab_size: usize, cfg: ModelConfig, rng: &mut impl Rng) -> Result<Self> {
        let mut state_dict = HashMap::new();
        let graph = Graph::with_arena_capacity(262_144);
        let normal = Normal::new(0.0, 0.08).context("failed to initialize normal distribution")?;

        let mut matrix = |nout: usize, nin: usize| {
            (0..nout)
                .map(|_| {
                    (0..nin)
                        .map(|_| graph.value(normal.sample(rng)))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };

        state_dict.insert("wte".to_string(), matrix(vocab_size, cfg.n_embd));
        state_dict.insert("wpe".to_string(), matrix(cfg.block_size, cfg.n_embd));
        state_dict.insert("lm_head".to_string(), matrix(vocab_size, cfg.n_embd));

        for i in 0..cfg.n_layer {
            state_dict.insert(
                format!("layer{}.attn_wq", i),
                matrix(cfg.n_embd, cfg.n_embd),
            );
            state_dict.insert(
                format!("layer{}.attn_wk", i),
                matrix(cfg.n_embd, cfg.n_embd),
            );
            state_dict.insert(
                format!("layer{}.attn_wv", i),
                matrix(cfg.n_embd, cfg.n_embd),
            );
            state_dict.insert(
                format!("layer{}.attn_wo", i),
                matrix(cfg.n_embd, cfg.n_embd),
            );
            state_dict.insert(
                format!("layer{}.mlp_fc1", i),
                matrix(4 * cfg.n_embd, cfg.n_embd),
            );
            state_dict.insert(
                format!("layer{}.mlp_fc2", i),
                matrix(cfg.n_embd, 4 * cfg.n_embd),
            );
        }

        Ok(GPT {
            state_dict,
            graph,
            cfg,
        })
    }

    pub fn from_weights(
        vocab_size: usize,
        cfg: ModelConfig,
        weights: HashMap<String, Vec<Vec<f64>>>,
    ) -> Result<Self> {
        let graph = Graph::with_arena_capacity(262_144);
        let mut state_dict = HashMap::new();

        if !weights.contains_key("wte")
            || !weights.contains_key("wpe")
            || !weights.contains_key("lm_head")
        {
            return Err(anyhow::anyhow!("checkpoint missing required tensors"));
        }

        for (name, matrix) in weights {
            let converted: Vec<Vec<Value>> = matrix
                .into_iter()
                .map(|row| row.into_iter().map(|v| graph.value(v)).collect())
                .collect();
            state_dict.insert(name, converted);
        }

        let wte_rows = state_dict
            .get("wte")
            .ok_or_else(|| NamegenError::MissingTensor("wte".to_string()))?
            .len();
        if wte_rows != vocab_size {
            return Err(anyhow::anyhow!(
                "checkpoint vocab mismatch: expected {vocab_size}, got {wte_rows}"
            ));
        }

        Ok(Self {
            state_dict,
            graph,
            cfg,
        })
    }

    pub fn params(&self) -> Vec<Value> {
        let mut params = Vec::new();
        let mut keys: Vec<_> = self.state_dict.keys().collect();
        keys.sort();
        for key in keys {
            for row in &self.state_dict[key] {
                for &p in row {
                    params.push(p);
                }
            }
        }
        params
    }
}

pub fn new_kv_cache(n_layer: usize) -> Vec<Vec<Vec<Value>>> {
    vec![Vec::new(); n_layer]
}

pub fn linear(graph: &Graph, x: &[Value], w: &[Vec<Value>]) -> Vec<Value> {
    w.iter().map(|row| graph.dot(row, x)).collect()
}

fn tensor<'a>(gpt: &'a GPT, key: &str) -> Result<&'a Vec<Vec<Value>>> {
    gpt.state_dict
        .get(key)
        .ok_or_else(|| NamegenError::MissingTensor(key.to_string()).into())
}

pub fn forward(
    gpt: &GPT,
    token_id: usize,
    pos_id: usize,
    keys: &mut [Vec<Vec<Value>>],
    values: &mut [Vec<Vec<Value>>],
) -> Result<Vec<Value>> {
    let graph = &gpt.graph;
    let tok_emb = tensor(gpt, "wte")?
        .get(token_id)
        .ok_or_else(|| anyhow::anyhow!("token index out of range: {token_id}"))?;
    let pos_emb = tensor(gpt, "wpe")?
        .get(pos_id)
        .ok_or_else(|| anyhow::anyhow!("position index out of range: {pos_id}"))?;
    let mut x: Vec<Value> = tok_emb
        .iter()
        .zip(pos_emb.iter())
        .map(|(&t, &p)| graph.add(t, p))
        .collect();
    x = graph.rmsnorm(&x);

    let head_dim = gpt.cfg.head_dim();
    for li in 0..gpt.cfg.n_layer {
        let x_residual = x.clone();
        x = graph.rmsnorm(&x);

        let q = linear(graph, &x, tensor(gpt, &format!("layer{}.attn_wq", li))?);
        let k = linear(graph, &x, tensor(gpt, &format!("layer{}.attn_wk", li))?);
        let v = linear(graph, &x, tensor(gpt, &format!("layer{}.attn_wv", li))?);

        keys[li].push(k.clone());
        values[li].push(v.clone());

        let head_outs: Vec<Vec<Value>> = (0..gpt.cfg.n_head)
            .into_par_iter()
            .map(|h| {
                let hs = h * head_dim;
                let q_h = &q[hs..hs + head_dim];

                let mut attn_logits = Vec::with_capacity(keys[li].len());
                for k_t in keys[li].iter() {
                    let k_h_t = &k_t[hs..hs + head_dim];
                    let logit = graph.dot(q_h, k_h_t);
                    let scaled_logit = graph.div(logit, graph.value((head_dim as f64).sqrt()));
                    attn_logits.push(scaled_logit);
                }

                let attn_weights = graph.softmax(&attn_logits);
                let mut head_out = Vec::with_capacity(head_dim);
                for j in 0..head_dim {
                    let mut term_vals = Vec::with_capacity(attn_weights.len());
                    for (t, &weight) in attn_weights.iter().enumerate() {
                        let val_node = values[li][t][hs + j];
                        term_vals.push(graph.mul(weight, val_node));
                    }
                    head_out.push(graph.sum(&term_vals));
                }
                head_out
            })
            .collect();

        let x_attn: Vec<Value> = head_outs.into_iter().flatten().collect();
        x = linear(
            graph,
            &x_attn,
            tensor(gpt, &format!("layer{}.attn_wo", li))?,
        );
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(&a, &b)| graph.add(a, b))
            .collect();

        let x_residual_mlp = x.clone();
        x = graph.rmsnorm(&x);
        x = linear(graph, &x, tensor(gpt, &format!("layer{}.mlp_fc1", li))?);
        x = x.into_iter().map(|xi| graph.relu(xi)).collect();
        x = linear(graph, &x, tensor(gpt, &format!("layer{}.mlp_fc2", li))?);
        x = x
            .iter()
            .zip(x_residual_mlp.iter())
            .map(|(&a, &b)| graph.add(a, b))
            .collect();
    }

    Ok(linear(graph, &x, tensor(gpt, "lm_head")?))
}
