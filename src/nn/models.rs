use crate::engine::Value;

pub trait Module {
    fn zero_grad(&self) -> ();

    fn parameters(&self) -> Vec<Value>;
}

pub struct Neuron();
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

pub struct Layer();
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

    pub fn forward(&self, data: &[Value]) -> Vec<Value> {
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

pub struct Trainer {
    model: Box<dyn Module>,
    epochs: i32,
}
impl Trainer {
    pub fn new<M: Module + 'static>(model: M, epochs: Option<i32>) -> Self {
        Trainer {
            model: Box::new(model),
            epochs: match epochs {
                None => 32,
                Some(x) => x,
            },
        }
    }
    pub fn fit(&mut self, data: &Vec<Vec<f64>>, labels: &[u8]) -> &mut Self {
        // Convert data and labels to Values as needed
        let data: Vec<Vec<Value>> = data.iter().map(|row| row.iter().map(|&x| Value::from(x)).collect()).collect();
        dbg!(&data, &labels);

        let loss = 0.0;
        let acc = 0.0;
        for e in 0..self.epochs {
            println!("Epoch {e}. \n\tloss: {loss}\n\tacc {acc}")
        }
        self
    }
}
