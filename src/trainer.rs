use crate::errors::NamegenError;
use crate::model::{BLOCK_SIZE, GPT, forward, new_kv_cache};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result};
use indicatif::ProgressBar;
use rand::Rng;
use rand::seq::SliceRandom;

#[derive(Debug, Clone, Copy)]
pub struct TrainOptions {
    pub epochs: usize,
    pub learning_rate: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainingSummary {
    pub final_loss: f64,
    pub steps: usize,
}

pub fn train_model(
    docs: &[String],
    tokenizer: &Tokenizer,
    gpt: &GPT,
    opts: TrainOptions,
    rng: &mut impl Rng,
    progress: Option<&ProgressBar>,
) -> Result<TrainingSummary> {
    if docs.is_empty() {
        return Err(NamegenError::EmptyDataset.into());
    }

    let params = gpt.params();
    let params_count = gpt.graph.nodes_count();
    let graph = &gpt.graph;

    let beta1 = 0.85;
    let beta2 = 0.99;
    let eps_adam = 1e-8;
    let mut m = vec![0.0; params.len()];
    let mut v = vec![0.0; params.len()];

    let total_steps = opts.epochs.saturating_mul(docs.len());
    let mut global_step = 0usize;
    let mut final_loss = 0.0;
    let mut indices: Vec<usize> = (0..docs.len()).collect();

    for _epoch in 0..opts.epochs {
        indices.shuffle(rng);
        for &doc_idx in &indices {
            let doc = &docs[doc_idx];
            graph.truncate(params_count);

            let mut tokens: Vec<usize> = vec![tokenizer.bos_token()];
            tokens.extend(tokenizer.encode(doc)?);
            tokens.push(tokenizer.bos_token());

            let n = BLOCK_SIZE.min(tokens.len().saturating_sub(1));
            if n == 0 {
                continue;
            }

            let mut keys = new_kv_cache(gpt.cfg.n_layer);
            let mut values = new_kv_cache(gpt.cfg.n_layer);
            let mut losses = Vec::with_capacity(n);

            for pos_id in 0..n {
                let token_id = tokens[pos_id];
                let target_id = tokens[pos_id + 1];
                let logits = forward(gpt, token_id, pos_id, &mut keys, &mut values)?;
                let probs = graph.softmax(&logits);

                let prob_target = probs[target_id];
                let log_p = graph.log(prob_target);
                let loss_t = graph.mul(log_p, graph.value(-1.0));
                losses.push(loss_t);
            }

            let total_loss = graph.sum(&losses);
            let loss = graph.div(total_loss, graph.value(n as f64));
            graph.backward(loss);
            final_loss = graph.node_data(loss);

            let lr_t = opts.learning_rate * (1.0 - global_step as f64 / total_steps.max(1) as f64);
            for (i, &p) in params.iter().enumerate() {
                let grad = graph.node_grad(p);
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad;
                v[i] = beta2 * v[i] + (1.0 - beta2) * grad * grad;
                let m_hat = m[i] / (1.0 - beta1.powi(global_step as i32 + 1));
                let v_hat = v[i] / (1.0 - beta2.powi(global_step as i32 + 1));
                graph.add_node_data(p, -(lr_t * m_hat / (v_hat.sqrt() + eps_adam)));
            }

            global_step += 1;
            if let Some(pb) = progress {
                pb.inc(1);
                pb.set_message(format!("loss {:.4}", final_loss));
            }
        }
    }

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    if !final_loss.is_finite() {
        return Err(anyhow::anyhow!("training diverged with non-finite loss"))
            .context("model training failed");
    }

    Ok(TrainingSummary {
        final_loss,
        steps: global_step,
    })
}
