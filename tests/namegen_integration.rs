use namegen::{
    ModelConfig, SamplingOptions, Tokenizer, forward, load_checkpoint, new_kv_cache,
    save_checkpoint,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn tokenizer_roundtrip_and_unknown_character() -> Result<(), Box<dyn Error>> {
    let docs = vec!["anna".to_string(), "maria".to_string(), "zoe".to_string()];
    let tokenizer = Tokenizer::from_docs_with_affixes(&docs, &[], &[])?;

    let text = "anna";
    let encoded = tokenizer.encode(text)?;
    let decoded: String = encoded
        .into_iter()
        .map(|token| tokenizer.decode_token(token))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .collect();

    assert_eq!(decoded, text);
    assert!(tokenizer.encode("anna$").is_err());
    Ok(())
}

#[test]
fn sampling_is_seed_deterministic_and_validates_inputs() -> Result<(), Box<dyn Error>> {
    let weights = [0.05, 0.15, 0.25, 0.55];
    let opts = SamplingOptions {
        temperature: 1.0,
        top_k: Some(3),
        top_p: Some(0.95),
    }
    .validate()?;

    let mut rng_a = StdRng::seed_from_u64(99);
    let mut rng_b = StdRng::seed_from_u64(99);

    let samples_a: Vec<usize> = (0..64)
        .map(|_| namegen::sampling::sample_index(&weights, opts, &mut rng_a))
        .collect::<Result<Vec<_>, _>>()?;
    let samples_b: Vec<usize> = (0..64)
        .map(|_| namegen::sampling::sample_index(&weights, opts, &mut rng_b))
        .collect::<Result<Vec<_>, _>>()?;

    assert_eq!(samples_a, samples_b);

    assert!(
        SamplingOptions {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        }
        .validate()
        .is_err()
    );
    assert!(
        SamplingOptions {
            temperature: 1.0,
            top_k: Some(0),
            top_p: None,
        }
        .validate()
        .is_err()
    );
    assert!(
        SamplingOptions {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.0),
        }
        .validate()
        .is_err()
    );

    Ok(())
}

#[test]
fn checkpoint_roundtrip_preserves_logits() -> Result<(), Box<dyn Error>> {
    let docs = vec!["anna".to_string(), "maria".to_string(), "zoe".to_string()];
    let tokenizer = Tokenizer::from_docs_with_affixes(&docs, &[], &[])?;
    let cfg = ModelConfig::default();

    let mut init_rng = StdRng::seed_from_u64(7);
    let gpt = namegen::GPT::new(tokenizer.vocab_size(), cfg, &mut init_rng)?;

    let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let mut path = std::env::temp_dir();
    path.push(format!(
        "namegen_test_ckpt_{}_{}.bin",
        std::process::id(),
        unique
    ));

    save_checkpoint(&path, &gpt, &tokenizer)?;
    let (loaded, loaded_tokenizer) = load_checkpoint(&path)?;
    std::fs::remove_file(&path)?;

    assert_eq!(loaded_tokenizer.vocab_size(), tokenizer.vocab_size());
    assert_eq!(loaded.cfg.n_embd, cfg.n_embd);
    assert_eq!(loaded.cfg.n_head, cfg.n_head);
    assert_eq!(loaded.cfg.n_layer, cfg.n_layer);
    assert_eq!(loaded.cfg.block_size, cfg.block_size);

    let mut keys_a = new_kv_cache(cfg.n_layer);
    let mut values_a = new_kv_cache(cfg.n_layer);
    let logits_a = forward(&gpt, tokenizer.bos_token(), 0, &mut keys_a, &mut values_a)?;

    let mut keys_b = new_kv_cache(cfg.n_layer);
    let mut values_b = new_kv_cache(cfg.n_layer);
    let logits_b = forward(
        &loaded,
        loaded_tokenizer.bos_token(),
        0,
        &mut keys_b,
        &mut values_b,
    )?;

    assert_eq!(logits_a.len(), logits_b.len());
    for (a, b) in logits_a.iter().zip(logits_b.iter()) {
        let da = gpt.graph.node_data(*a);
        let db = loaded.graph.node_data(*b);
        assert!((da - db).abs() < 1e-12, "logit mismatch: {da} vs {db}");
    }

    Ok(())
}
