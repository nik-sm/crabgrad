use crate::engine::Value;
use crate::nn::cross_entropy_single;
use crate::optim::{ADAM, Optim, OptimType, SGD};
use crate::utils::try_init_logging;
use anyhow::{Result, bail};
use itertools::Itertools;
use log::Level;
use rand::rng;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};

pub trait Module {
    fn zero_grad(&self) {
        for param in self.parameters() {
            let mut param = param.borrow_mut();
            param.grad = None
        }
    }

    fn parameters(&self) -> Vec<&Value>;

    fn forward(&self, data: &Vec<Value>) -> Result<Vec<Value>>;
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
}

impl Module for Neuron {
    fn forward(&self, data: &Vec<Value>) -> Result<Vec<Value>> {
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
}

impl Module for Layer {
    fn forward(&self, data: &Vec<Value>) -> Result<Vec<Value>> {
        self.neurons.iter().map(|neuron| neuron.forward(&data)).collect()
    }

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
}

impl Module for MLP {
    fn forward(&self, data: &Vec<Value>) -> Result<Vec<Value>> {
        let mut result = data;
        for layer in self.layers.iter() {
            result = layer.forward(result)?;
        }
        Ok(result)
    }

    fn parameters(&self) -> Vec<&Value> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}

pub struct Trainer<'a> {
    model: &'a dyn Module,
    optim: &'a dyn Optim,
    epochs: usize,
    batch_size: usize,
}
impl<'a> Trainer<'a> {
    pub fn new(model: &'a impl Module, epochs: usize, optim: &'a impl Optim, batch_size: usize) -> Self {
        Trainer { model, epochs, optim, batch_size }
    }

    pub fn fit(&mut self, mut data_labels: Vec<(Vec<f64>, i64)>) -> Result<()> {
        // Convert data and labels to Values as needed
        try_init_logging()?;

        let mut rng = rng();
        for e in 0..self.epochs {
            data_labels.shuffle(&mut rng);
            log::info!("{}", format!("{:-^10}!", format!("Epoch {e}")));
            for (batch_idx, chunk) in data_labels.iter().chunks(self.batch_size).into_iter().enumerate() {
                log::info!("Batch {batch_idx}");

                self.optim.zero_grad();

                let mut loss = Value::from(0.0);
                for (data, label) in chunk {
                    let data: Vec<Value> = data.iter().map(Value::from).collect();
                    let label: Value = Value::from(label);
                    let logits: Vec<Value> = self.model.forward(&data)?;
                    loss = loss + cross_entropy_single(&label, &logits);
                }
                loss.backward();

                self.optim.step();
            }
        }

        Ok(())
    }
}
