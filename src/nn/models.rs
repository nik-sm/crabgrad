use crate::engine::Value;

trait Module {
    fn zero_grad(&self) -> ();

    fn parameters(&self) -> Vec<Value>;
}

struct Neuron();
impl Neuron {
    fn new() -> Self {
        unimplemented!()
    }
}
impl Module for Neuron {
    fn zero_grad(&self) -> () {
        unimplemented!()
    }
    fn parameters(&self) -> Vec<Value> {
        unimplemented!()
    }
}

struct Layer();
impl Layer {
    fn new() -> Self {
        unimplemented!()
    }
}
impl Module for Layer {
    fn zero_grad(&self) -> () {
        unimplemented!()
    }
    fn parameters(&self) -> Vec<Value> {
        unimplemented!()
    }
}

pub struct MLP();
impl MLP {
    pub fn new() -> MLP {
        unimplemented!()
    }

    pub fn forward(&self, data: Vec<Value>) -> Vec<Value> {
        unimplemented!()
    }
}
impl Module for MLP {
    fn zero_grad(&self) -> () {
        unimplemented!()
    }
    fn parameters(&self) -> Vec<Value> {
        unimplemented!()
    }
}
