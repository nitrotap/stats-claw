//! Maximum-likelihood estimation: the generic MLE framework plus the concrete
//! per-distribution likelihood models built on top of it.
//!
//! A parametric model implements [`LogLikelihood`] and defers optimization to
//! [`fit_mle`], which maximizes `ℓ(θ; data)` by minimizing `−ℓ` with the
//! framework's L-BFGS optimizer and reports the fit as an [`MleFit`]. The
//! generic machinery lives in [`mle`]; concrete likelihood families are added as
//! sibling submodules that reuse it.

pub mod types;
pub use types::*;
pub mod binomial;
pub mod categorical;
pub mod exponential;
pub mod mle;
pub mod normal;
pub mod poisson;

pub use mle::{LogLikelihood, MleFit, fit_mle};
