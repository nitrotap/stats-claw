//! Typed error and `Result` alias for the framework's fallible numerics.
//!
//! This module defines the crate-wide `Error` enum and the `Result<T>` alias
//! that all hand-written numerics return on bad input or domain violations
//! (e.g. a negative scale parameter, an empty sample). It is the single place
//! recoverable failures are modelled, so call sites never reach for `unwrap`.

use std::fmt;

/// A recoverable failure raised by the framework's hand-written numerics.
///
/// Numerics never panic on bad input — they return one of these variants so the
/// caller can decide how to recover. The message-carrying variants embed a short
/// human-readable reason; the unit variants are self-describing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A required sample or collection was empty.
    EmptyInput,
    /// A parameter or value violated a precondition; carries the reason.
    InvalidInput(String),
    /// The input is technically valid but degenerate (e.g. zero variance);
    /// carries the reason.
    DegenerateInput(String),
    /// There were too few observations to compute the requested quantity.
    InsufficientData,
    /// An iterative procedure exhausted its budget without converging.
    NotConverged,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input was empty"),
            Self::InvalidInput(m) => write!(f, "invalid input: {m}"),
            Self::DegenerateInput(m) => write!(f, "degenerate input: {m}"),
            Self::InsufficientData => write!(f, "insufficient data"),
            Self::NotConverged => write!(f, "did not converge"),
        }
    }
}

impl std::error::Error for Error {}

/// Result alias for the framework's fallible numerics, defaulting the error to
/// [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_human_readable() {
        assert_eq!(
            Error::EmptyInput.to_string(),
            "input was empty",
            "EmptyInput display text changed"
        );
    }

    #[test]
    fn display_includes_detail_for_message_variants() {
        assert_eq!(
            Error::InvalidInput("scale must be > 0".to_owned()).to_string(),
            "invalid input: scale must be > 0"
        );
        assert_eq!(
            Error::DegenerateInput("zero variance".to_owned()).to_string(),
            "degenerate input: zero variance"
        );
    }

    #[test]
    fn usable_as_std_error() {
        fn boxed() -> Box<dyn std::error::Error> {
            Box::new(Error::NotConverged)
        }
        assert_eq!(boxed().to_string(), "did not converge");
    }
}
