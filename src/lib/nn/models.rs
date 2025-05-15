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

#[derive(Debug)]
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

    fn normalize(&mut self) {
        // TODO - this approach was extremely slow - why?
        // let norm = norm(&self.weights);
        // self.weights = self.weights.iter().map(|w| w / norm.clone()).collect();

        let norm = self.weights.iter().fold(0.0, |acc, val| acc + val.data().powf(2.0)).sqrt();
        self.weights.iter().for_each(|p| p.borrow_mut().data = p.data() / norm);
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

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}
impl Layer {
    fn new(in_dim: usize, out_dim: usize, bias: bool, relu: bool) -> Self {
        Self { neurons: (0..out_dim).map(|_| Neuron::new(in_dim, bias, relu)).collect() }
    }

    fn normalize(&mut self) {
        for neuron in &mut self.neurons {
            neuron.normalize();
        }
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

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}
impl MLP {
    pub fn new(in_dim: usize, hidden_dims: Vec<usize>, out_dim: usize, bias: bool) -> Self {
        // API ensures at least one layer

        // For all_dims.len() == n, always n-1 layers total
        let n = &hidden_dims.len() + 1;
        let all_dims = std::iter::once(in_dim).chain(hidden_dims).chain(std::iter::once(out_dim));

        let mut layers: Vec<Layer> = vec![];

        for (idx, (d1, d2)) in all_dims.tuple_windows().enumerate() {
            let relu = idx < n;
            layers.push(Layer::new(d1, d2, bias, relu))
        }

        Self { layers }
    }

    fn normalize(&mut self) {
        for layer in &mut self.layers {
            layer.normalize();
        }
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
        Ok(next)
    }

    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{norm, sum};
    use crate::{Optim, optim::SGD};
    use crate::{assert_close, assert_not_close};
    use anyhow::Result;

    #[test]
    fn test_neuron_normalize() {
        let mut n = Neuron::new(3, false, false);
        assert_not_close!(norm(&n.weights).data(), 1.0);
        n.normalize();
        assert_close!(norm(&n.weights).data(), 1.0);
    }

    #[test]
    fn test_layer_normalize() {
        let mut layer = Layer::new(3, 2, false, false);
        for n in &layer.neurons {
            assert_not_close!(norm(&n.weights).data(), 1.0);
        }
        layer.normalize();
        for n in &layer.neurons {
            assert_close!(norm(&n.weights).data(), 1.0);
        }
    }

    #[test]
    fn test_mlp_normalize() {
        let mut mlp = MLP::new(3, vec![3, 2], 2, false);
        for layer in &mlp.layers {
            for n in &layer.neurons {
                assert_not_close!(norm(&n.weights).data(), 1.0);
            }
        }
        mlp.normalize();
        for layer in &mlp.layers {
            for n in &layer.neurons {
                assert_close!(norm(&n.weights).data(), 1.0);
            }
        }
    }

    #[test]
    fn test_neuron_sgd() -> Result<()> {
        // Check that a single neuron will move as expected
        // We multiply with a fixed vector, and keep re-normalizing the layer weights to a unit vector
        // The neuron's weights should move towards being aligned
        let mut n = Neuron::new(3, false, false);
        // n.normalize();

        let x = vec![Value::from(1.0), Value::from(0.0), Value::from(0.0)];

        let optim = SGD::new(n.parameters(), 0.1);

        let before = n.parameters();
        assert_not_close!(before[0].data(), 1.0);
        assert_not_close!(before[1].data(), 0.0);
        assert_not_close!(before[2].data(), 0.0);

        for _ in 0..1000 {
            // println!("{i}");
            // Move param vector towards being a unit vector aligned with the data
            let out = n.forward(&x)?.get(0).unwrap().to_owned();
            let loss = 1 - out;

            optim.zero_grad();
            loss.backward();
            optim.step();

            // Normalize to unit length
            n.normalize();
        }

        let after = n.parameters();

        assert_close!(after[0].data(), 1.0);
        assert_close!(after[1].data(), 0.0);
        assert_close!(after[2].data(), 0.0);

        Ok(())
    }

    #[test]
    fn test_layer_sgd() -> Result<()> {
        // Check that each neuron in a single layer will move as expected
        // Same strategy as used for single neuron case
        let mut layer = Layer::new(3, 2, false, false);
        layer.normalize();

        let x = vec![Value::from(1.0), Value::from(0.0), Value::from(0.0)];

        let optim = SGD::new(layer.parameters(), 0.1);

        let neuron_0 = &layer.neurons[0];
        assert_not_close!(neuron_0.parameters()[0].data(), 1.0);
        assert_not_close!(neuron_0.parameters()[1].data(), 0.0);
        assert_not_close!(neuron_0.parameters()[2].data(), 0.0);

        let neuron_1 = &layer.neurons[1];
        assert_not_close!(neuron_1.parameters()[0].data(), 1.0);
        assert_not_close!(neuron_1.parameters()[1].data(), 0.0);
        assert_not_close!(neuron_1.parameters()[2].data(), 0.0);

        for _ in 0..1000 {
            // Move param vector towards being a unit vector aligned with the data
            let out = layer.forward(&x)?;
            let loss = 1 - sum(&out);

            optim.zero_grad();
            loss.backward();
            optim.step();

            // Normalize to unit length
            layer.normalize();
        }

        let neuron_0 = &layer.neurons[0];
        dbg!(&neuron_0);
        assert_close!(neuron_0.parameters()[0].data(), 1.0);
        assert_close!(neuron_0.parameters()[1].data(), 0.0);
        assert_close!(neuron_0.parameters()[2].data(), 0.0);

        let neuron_1 = &layer.neurons[1];
        dbg!(&neuron_1);
        assert_close!(neuron_1.parameters()[0].data(), 1.0);
        assert_close!(neuron_1.parameters()[1].data(), 0.0);
        assert_close!(neuron_1.parameters()[2].data(), 0.0);

        Ok(())
    }
}
