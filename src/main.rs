use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info};
use namegen::{
    AppConfig, ModelConfig, QualityFilter, SamplingOptions, Tokenizer, TrainDefaults, TrainOptions,
    forward, load_checkpoint, new_kv_cache, save_checkpoint, train_model,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "namegen", about = "Train and generate realistic random names")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Train {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        epochs: Option<usize>,
        #[arg(long = "learning-rate")]
        learning_rate: Option<f64>,
        #[arg(long = "save-model")]
        save_model: PathBuf,
        #[arg(long)]
        seed: Option<u64>,
    },
    Gen {
        #[arg(long, default_value_t = 20)]
        count: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long = "load-model")]
        load_model: Option<PathBuf>,
        #[arg(long = "top-k")]
        top_k: Option<usize>,
        #[arg(long = "top-p")]
        top_p: Option<f64>,
    },
}

fn merged_defaults(defaults: &TrainDefaults) -> (usize, f64) {
    (
        defaults.epochs.unwrap_or(2),
        defaults.learning_rate.unwrap_or(0.01),
    )
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env().init();
    let cli = Cli::parse();
    let config = AppConfig::load_if_exists(std::path::Path::new("Config.toml"))?;

    match cli.command {
        Commands::Train {
            dataset,
            epochs,
            learning_rate,
            save_model,
            seed,
        } => {
            let (def_epochs, def_lr) = merged_defaults(&config.defaults);
            let train_epochs = epochs.unwrap_or(def_epochs);
            let train_lr = learning_rate.unwrap_or(def_lr);

            let dataset_contents = std::fs::read_to_string(&dataset)
                .with_context(|| format!("failed to read dataset {}", dataset.display()))?;
            let (tokenizer, docs) = Tokenizer::from_lines(&dataset_contents)?;

            let model_cfg = ModelConfig::default();
            let seed_value = seed.unwrap_or(42);
            let mut rng = StdRng::seed_from_u64(seed_value);
            info!("training with seed={seed_value} docs={}", docs.len());

            let gpt = namegen::GPT::new(tokenizer.vocab_size(), model_cfg, &mut rng)?;
            let total_steps = train_epochs.saturating_mul(docs.len()) as u64;
            let pb = ProgressBar::new(total_steps);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .context("failed to build progress style")?,
            );

            let summary = train_model(
                &docs,
                &tokenizer,
                &gpt,
                TrainOptions {
                    epochs: train_epochs,
                    learning_rate: train_lr,
                },
                &mut rng,
                Some(&pb),
            )?;

            save_checkpoint(&save_model, &gpt, &tokenizer)?;
            println!(
                "{} {} steps={} final_loss={:.4}",
                "saved model:".green().bold(),
                save_model.display(),
                summary.steps,
                summary.final_loss
            );
        }
        Commands::Gen {
            count,
            temperature,
            seed,
            load_model,
            top_k,
            top_p,
        } => {
            let opts = SamplingOptions {
                temperature,
                top_k,
                top_p,
            }
            .validate()?;

            let model_path = load_model.unwrap_or_else(|| PathBuf::from("assets/pretrained.bin"));
            let (gpt, tokenizer) = load_checkpoint(&model_path)
                .with_context(|| format!("failed loading model {}", model_path.display()))?;

            let mut rng = StdRng::seed_from_u64(seed.unwrap_or(42));
            let mut seen = HashSet::new();
            let quality = QualityFilter::default();
            let params_count = gpt.graph.nodes_count();

            println!("{}", "generated names".bright_blue().bold());
            for idx in 0..count {
                let mut keys = new_kv_cache(gpt.cfg.n_layer);
                let mut values = new_kv_cache(gpt.cfg.n_layer);
                let mut token_id = tokenizer.bos_token();
                let mut chars = Vec::new();

                for pos_id in 0..gpt.cfg.block_size {
                    gpt.graph.truncate(params_count);
                    let logits = forward(&gpt, token_id, pos_id, &mut keys, &mut values)?;
                    let scaled_logits: Vec<_> = logits
                        .iter()
                        .map(|&l| gpt.graph.div(l, gpt.graph.value(opts.temperature)))
                        .collect();
                    let probs = gpt.graph.softmax(&scaled_logits);
                    let weights: Vec<f64> = probs.iter().map(|&p| gpt.graph.node_data(p)).collect();
                    token_id = namegen::sampling::sample_index(&weights, opts, &mut rng)
                        .context("sampling failed")?;

                    if token_id == tokenizer.bos_token() {
                        break;
                    }
                    chars.push(tokenizer.decode_token(token_id)?);
                }

                let candidate: String = chars.iter().collect();
                if quality.is_valid(&candidate, &seen) {
                    seen.insert(candidate.clone());
                    println!("{:>3}. {}", idx + 1, candidate.bright_magenta());
                } else {
                    debug!("filtered candidate '{}'", candidate);
                }
            }

            if seen.is_empty() {
                return Err(anyhow!("no names passed the quality filter"));
            }
        }
    }

    Ok(())
}
