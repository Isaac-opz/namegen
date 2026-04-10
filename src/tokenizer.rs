use crate::errors::NamegenError;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokenizer {
    chars: Vec<char>,
    bos_token: usize,
    #[serde(skip)]
    index: HashMap<char, usize>,
}

impl Tokenizer {
    pub fn from_docs_with_affixes(
        docs: &[String],
        prefixes: &[String],
        suffixes: &[String],
    ) -> Result<Self> {
        if docs.is_empty() {
            return Err(NamegenError::EmptyDataset.into());
        }

        let mut chars: Vec<char> = docs
            .iter()
            .flat_map(|d| d.chars())
            .filter(|c| c.is_alphabetic() || *c == '-' || *c == '\'')
            .collect();
        for p in prefixes {
            chars.extend(p.chars());
        }
        for s in suffixes {
            chars.extend(s.chars());
        }

        chars.sort_unstable();
        chars.dedup();

        let bos_token = chars.len();
        let mut tk = Self {
            chars,
            bos_token,
            index: HashMap::new(),
        };
        tk.rebuild_index();
        Ok(tk)
    }

    pub fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, ch) in self.chars.iter().enumerate() {
            self.index.insert(*ch, i);
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        text.chars()
            .map(|c| {
                self.index
                    .get(&c)
                    .copied()
                    .ok_or_else(|| NamegenError::UnknownCharacter(c).into())
            })
            .collect()
    }

    pub fn decode_token(&self, token: usize) -> Result<char> {
        self.chars
            .get(token)
            .copied()
            .ok_or_else(|| NamegenError::InvalidToken(token).into())
    }

    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    pub fn bos_token(&self) -> usize {
        self.bos_token
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len() + 1
    }

    pub fn from_lines(contents: &str) -> Result<(Self, Vec<String>)> {
        let docs: Vec<String> = contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect();

        let tokenizer = Self::from_docs_with_affixes(&docs, &[], &[])
            .context("failed to build tokenizer from dataset")?;
        Ok((tokenizer, docs))
    }
}
