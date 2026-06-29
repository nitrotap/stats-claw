//! Second-order optimizers: Newton's method (Hessian-based) and L-BFGS (limited-
//! memory quasi-Newton).
//!
//! Both use curvature information to take better-scaled steps than first-order
//! methods. Newton solves the Newton system with an inner conjugate-gradient
//! loop (the `Newton-CG` strategy); L-BFGS approximates the inverse Hessian from
//! a short history of gradient/step pairs via the two-loop recursion.

pub mod lbfgs;
pub mod newton;

pub use lbfgs::lbfgs;
pub use newton::newton;
