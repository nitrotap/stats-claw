//! First-order optimizers: learning-rate methods (gradient descent, SGD, Adam,
//! `RMSProp`, `AdaGrad`) and the Fletcher–Reeves conjugate gradient.
//!
//! Each optimizer drives an [`Objective`](super::Objective) toward a minimum and
//! returns an [`OptimizeResult`](super::OptimizeResult). The learning-rate
//! methods stop when the gradient norm falls below a tolerance; conjugate
//! gradient additionally restarts every `n` steps.

pub mod adagrad;
pub mod adam;
pub mod conjugate_gradient;
pub mod gradient_descent;
pub mod rmsprop;
pub mod sgd;

pub use adagrad::adagrad;
pub use adam::adam;
pub use conjugate_gradient::conjugate_gradient;
pub use gradient_descent::gradient_descent;
pub use rmsprop::rmsprop;
pub use sgd::sgd;
