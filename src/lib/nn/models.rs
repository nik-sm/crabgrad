use crate::{argmax, engine::Value};
use anyhow::{Result, bail};
use itertools::Itertools;
use rand_distr::{Distribution, Normal};

pub trait Module {
    fn zero_grad(&self) {
        for param in self.parameters() {
            let mut param = param.borrow_mut();
            param.grad = None
        }
    }

    fn parameters(&self) -> Vec<Value>;

    fn forward(&self, data: &[Value]) -> Result<Vec<Value>>;
}

pub trait Classifier: Module {
    fn score(&self, data_labels: &[(Vec<Value>, usize)]) -> Result<f64> {
        let mut n_correct = 0usize;
        let mut n_total = 0usize;

        for (data, label) in data_labels.iter() {
            let logits = self.forward(data)?;
            let pred = argmax(&logits);
            n_total += 1;
            if pred == *label as usize {
                n_correct += 1;
            }
        }

        Ok(n_correct as f64 / n_total as f64)
    }
}
impl Classifier for MLP {}

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
}

impl Module for Neuron {
    fn forward(&self, data: &[Value]) -> Result<Vec<Value>> {
        if self.weights.len() != data.len() {
            bail!("shape mismatch")
        }
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
        Ok(vec![result])
    }

    fn parameters(&self) -> Vec<Value> {
        self.weights.iter().chain(self.bias.as_ref()).cloned().collect()
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}
impl Layer {
    fn new(in_dim: usize, out_dim: usize, bias: bool, relu: bool) -> Self {
        Self { neurons: (0..out_dim).map(|_| Neuron::new(in_dim, bias, relu)).collect() }
    }
}

impl Module for Layer {
    fn forward(&self, data: &[Value]) -> Result<Vec<Value>> {
        self.neurons.iter().map(|neuron| neuron.forward(data)).flatten_ok().collect()
    }

    fn parameters(&self) -> Vec<Value> {
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
            let relu = idx < n;
            layers.push(Layer::new(d1, d2, bias, relu))
        }

        Self { layers }
    }
}

impl Module for MLP {
    fn forward(&self, data: &[Value]) -> Result<Vec<Value>> {
        let mut prev: &[Value] = data;
        let mut next: Vec<Value> = vec![];
        for layer in &self.layers {
            next = layer.forward(prev)?;
            prev = &next;
        }
        Ok(next.clone())
    }

    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}
