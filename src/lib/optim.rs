use crate::engine::Value;

pub trait Optim {
    fn zero_grad(&self);
    fn step(&self);
}

pub enum OptimType {
    SGD,
    ADAM,
}

pub struct SGD {
    parameters: Vec<Value>,
    lr: f64,
}

impl SGD {
    pub fn new(parameters: Vec<Value>, lr: f64) -> Self {
        Self { parameters, lr }
    }
}

impl Optim for SGD {
    fn zero_grad(&self) {
        for p in &self.parameters {
            p.zero_grad();
        }
    }
    fn step(&self) {
        for p in &self.parameters {
            match p.backward_fn() {
                Some(_) => {
                    dbg!("before", &p);
                    p.borrow_mut().data = p.data() - self.lr * p.grad().expect("step without grad");
                    dbg!("after", &p);
                    panic!("disco")
                }
                _ => p.borrow_mut().data = p.data() - self.lr * p.grad().expect("step without grad"),
            }
        }
    }
}
