use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct QualityFilter {
    pub min_len: usize,
    pub max_len: usize,
    pub alpha_only: bool,
}

impl Default for QualityFilter {
    fn default() -> Self {
        Self {
            min_len: 3,
            max_len: 14,
            alpha_only: true,
        }
    }
}

impl QualityFilter {
    pub fn is_valid(&self, candidate: &str, seen: &HashSet<String>) -> bool {
        if candidate.len() < self.min_len || candidate.len() > self.max_len {
            return false;
        }
        if self.alpha_only && !candidate.chars().all(|c| c.is_alphabetic()) {
            return false;
        }
        !seen.contains(candidate)
    }
}
