pub mod loss;
pub use loss::{cross_entropy, cross_entropy_single};

pub mod models;
pub use models::{MLP, Module};

pub mod trainer;
pub use trainer::Trainer;
