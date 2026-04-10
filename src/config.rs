use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct TrainDefaults {
    pub epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub temperature: Option<f64>,
    pub count: Option<usize>,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct AppConfig {
    pub defaults: TrainDefaults,
}

impl AppConfig {
    pub fn load_if_exists(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config file {}", path.display()))?;
        let cfg: AppConfig = toml::from_str(&raw)
            .with_context(|| format!("failed to parse config file {}", path.display()))?;
        Ok(cfg)
    }
}
