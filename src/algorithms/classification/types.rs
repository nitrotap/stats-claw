//! Plain-data parameter structs for the library's statistical constructs.
//!
//! Each struct carries only the parameters (and any descriptive string fields)
//! that identify a construct; all numerics live in the behaviour traits and
//! inherent methods implemented for these types in the sibling modules. The
//! structs derive `Default` so callers build them with struct-update syntax and
//! set only the fields they care about.
//!
//! This file is produced mechanically by the `carve` tool from the source
//! project; edit the carve inputs rather than this file.

/// Result of a classification model evaluation.
#[derive(Debug, Clone, Default)]
pub struct ClassificationResult {
    /// Classification accuracy.
    pub accuracy: f64,
    /// Classification precision.
    pub precision: f64,
    /// Classification recall.
    pub recall: f64,
    /// F1 score.
    pub f1_score: f64,
    /// Unique identifier for a result.
    pub result_id: String,
    /// Time a result was produced.
    pub timestamp: String,
    /// Free-text description.
    pub description: String,
}
