pub mod engine;
pub mod nn;
pub mod util;

pub use engine::Value;

pub enum Node {
    Literal(Value),
    Add(Box<Node>, Box<Node>),
    Sub(Box<Node>, Box<Node>),
    Mul(Box<Node>, Box<Node>),
    Div(Box<Node>, Box<Node>),
}
impl Node {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let a = Value::new(2.0);
        let b = Value::new(2.0);

        let c = a + b;
    }
}
