use thiserror::Error;

#[derive(Debug, Error)]
pub enum NamegenError {
    #[error("character not in tokenizer vocabulary: {0}")]
    UnknownCharacter(char),
    #[error("invalid token id: {0}")]
    InvalidToken(usize),
    #[error("invalid temperature {0}; expected range 0.1..=2.0")]
    InvalidTemperature(f64),
    #[error("invalid top-p {0}; expected range (0.0, 1.0]")]
    InvalidTopP(f64),
    #[error("invalid top-k {0}; expected value > 0")]
    InvalidTopK(usize),
    #[error("empty dataset")]
    EmptyDataset,
    #[error("model tensor missing: {0}")]
    MissingTensor(String),
}
