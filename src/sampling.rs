use crate::errors::NamegenError;
use anyhow::{Context, Result};
use rand::Rng;
use rand::distr::{Distribution, weighted::WeightedIndex};

#[derive(Debug, Clone, Copy)]
pub struct SamplingOptions {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
}

impl SamplingOptions {
    pub fn validate(self) -> Result<Self> {
        if !(0.1..=2.0).contains(&self.temperature) {
            return Err(NamegenError::InvalidTemperature(self.temperature).into());
        }
        if let Some(k) = self.top_k
            && k == 0
        {
            return Err(NamegenError::InvalidTopK(k).into());
        }
        if let Some(p) = self.top_p
            && (!(0.0..=1.0).contains(&p) || p == 0.0)
        {
            return Err(NamegenError::InvalidTopP(p).into());
        }
        Ok(self)
    }
}

pub fn sample_index(weights: &[f64], opts: SamplingOptions, rng: &mut impl Rng) -> Result<usize> {
    let opts = opts.validate()?;

    let mut candidates: Vec<(usize, f64)> = weights
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, w)| *w > 0.0 && w.is_finite())
        .collect();

    if candidates.is_empty() {
        return Err(anyhow::anyhow!("no valid sampling candidates"));
    }

    if let Some(k) = opts.top_k {
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        candidates.truncate(k.min(candidates.len()));
    }

    if let Some(p) = opts.top_p {
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut cum = 0.0;
        let mut kept = Vec::new();
        for c in candidates {
            cum += c.1;
            kept.push(c);
            if cum >= p {
                break;
            }
        }
        candidates = kept;
    }

    let norm: f64 = candidates.iter().map(|(_, w)| *w).sum();
    let normalized: Vec<f64> = if norm > 0.0 {
        candidates.iter().map(|(_, w)| *w / norm).collect()
    } else {
        vec![1.0 / candidates.len() as f64; candidates.len()]
    };

    let dist = WeightedIndex::new(&normalized).context("failed to build weighted index")?;
    let sampled = dist.sample(rng);
    Ok(candidates[sampled].0)
}
