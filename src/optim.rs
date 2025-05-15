use crate::nn::Module;
pub trait Optim {
    fn zero_grad(&self);
    fn step(&self);
}

pub enum OptimType {
    SGD,
    ADAM,
}

pub struct SGD<'a> {
    model: &'a dyn Module,
    lr: f64,
}

impl<'a> SGD<'a> {
    pub fn new(model: &'a dyn Module, lr: f64) -> Self {
        Self { model, lr }
    }
}

impl Optim for SGD<'_> {
    fn zero_grad(&self) {
        todo!()
    }
    fn step(&self) {
        todo!()
    }
}

pub struct ADAM<'a> {
    model: &'a dyn Module,
    lr: f64,
}

impl<'a> ADAM<'a> {
    pub fn new(model: &'a dyn Module, lr: f64) -> Self {
        Self { model, lr }
    }
}

impl Optim for ADAM<'_> {
    fn zero_grad(&self) {
        todo!()
    }
    fn step(&self) {
        todo!()
    }
}
