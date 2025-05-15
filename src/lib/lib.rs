pub mod engine;
pub use engine::{DiscreteLabel, FloatDataScalar, IntDataScalar, Value, argmax, norm, pow, prod, sum};

pub mod nn;

pub mod ops;
pub mod utils;

pub mod optim;
pub use optim::{Optim, SGD};
