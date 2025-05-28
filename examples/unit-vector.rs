use anyhow::Result;
use crabgrad::engine::{sum, to_vec, FloatDataScalar, Value};
use crabgrad::nn::{Layer, Module};
use crabgrad::optim::{Optim, SGD};
use crabgrad::{assert_close, assert_vec_close};

fn main() -> Result<()> {
    fn get_weights(layer: &Layer) -> Vec<FloatDataScalar> {
        let mut weights = vec![];
        for neuron in &layer.neurons {
            weights.extend_from_slice(to_vec(&neuron.weights).as_slice());
        }
        weights
    }

    let mut layer = Layer::new(3, 1, false, false);
    let data = vec![Value::from(1.0), Value::from(0.0), Value::from(0.0)];
    let mut optim = SGD::new(&layer.parameters(), 0.1);

    dbg!("before", get_weights(&layer));

    for _ in 0..1000 {
        // Move param vector towards being a unit vector aligned with the data
        let out = layer.forward(&data)?;
        let loss = 1.0 - sum(&out);

        optim.zero_grad();
        loss.backward();
        optim.step();

        // Normalize to unit length
        layer.normalize();
    }

    dbg!("after", get_weights(&layer));

    let mut final_weights = vec![];
    for neuron in &layer.neurons {
        final_weights.extend_from_slice(neuron.weights.as_slice());
    }
    assert_vec_close!(final_weights, data);

    Ok(())
}
