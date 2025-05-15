use crate::engine::Value;
use itertools::Itertools;
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub trait Module {
    fn zero_grad(&self) {
        for param in self.parameters() {
            let mut param = param.borrow_mut();
            param.grad = None
        }
    }

    // TODO - how can we put mandatory .forward into this trait? Input for different level of
    // abstraction have different types (vec vs vec<vec>, etc)

    fn parameters(&self) -> Vec<&Value>;
}

pub struct Neuron {
    weights: Vec<Value>,
    bias: Option<Value>,
    relu: bool,
}
impl Neuron {
    fn new(in_dim: usize, bias: bool, relu: bool) -> Self {
        let mut rng = rand::rng();
        let gaussian = Normal::new(0.0, 1.0).expect("create gaussian");
        let weights: Vec<Value> = gaussian.sample_iter(&mut rng).take(in_dim).map(Value::from).collect();
        let bias = if bias { Some(Value::from(gaussian.sample(&mut rng))) } else { None };
        Self { weights, bias, relu }
    }

    fn forward(&self, data: &Vec<Value>) -> Value {
        assert!(self.weights.len() == data.len(), "shape mismatch");
        let mut result = self
            .weights
            .iter()
            .zip(data)
            .map(|(wi, xi)| wi.clone() * xi.clone()) // TODO - avoid the cloning, even if lightweight?
            .fold(Value::from(0.0), |acc, value| acc + value);
        if let Some(b) = &self.bias {
            result = result + b.clone();
        }
        if self.relu {
            result = result.relu()
        }
        result
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<&Value> {
        self.weights.iter().chain(self.bias.as_ref().into_iter()).collect()
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}
impl Layer {
    fn new(in_dim: usize, out_dim: usize, bias: bool, relu: bool) -> Self {
        Self { neurons: (0..out_dim).map(|_| Neuron::new(in_dim, bias, relu)).collect() }
    }

    fn forward(&self, data: Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|neuron| neuron.forward(&data)).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<&Value> {
        self.neurons.iter().flat_map(|neuron| neuron.parameters()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}
impl MLP {
    pub fn new(in_dim: usize, hidden_dims: Vec<usize>, out_dim: usize, bias: bool) -> Self {
        // API ensures at least one layer
        let n = &hidden_dims.len() + 1;
        let all_dims = std::iter::once(in_dim).chain(hidden_dims).chain(std::iter::once(out_dim));

        let mut layers: Vec<Layer> = vec![];

        // For all_dims.len() == n, always n-1 layers total
        for (idx, (d1, d2)) in all_dims.tuple_windows().enumerate() {
            let relu = if idx < n { true } else { false };
            layers.push(Layer::new(d1, d2, bias, relu))
        }

        Self { layers }
    }

    pub fn forward(&self, data: Vec<Value>) -> Vec<Value> {
        let mut result = data;
        for layer in self.layers.iter() {
            result = layer.forward(result);
        }
        result
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<&Value> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}

pub struct Trainer {
    model: Box<dyn Module>,
    epochs: usize,
}
impl Trainer {
    pub fn new<M: Module + 'static>(model: M, epochs: usize) -> Self {
        Trainer { model: Box::new(model), epochs }
    }

    pub fn fit(&mut self, labeled_data: impl Iterator<Item = (u8, Vec<f64>)>) -> &mut Self {
        // Convert data and labels to Values as needed
        let loss = 0.0;
        let acc = 0.0;
        for e in 0..self.epochs {
            println!("Epoch {e}. \n\tloss: {loss}\n\tacc {acc}")
        }
        todo!(); // Left off here. Need: optimizer, loss function, and step
        self
    }
}
