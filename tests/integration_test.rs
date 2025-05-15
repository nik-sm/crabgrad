use micrograd_rs::engine::Value;
use micrograd_rs::nn::MLP;
use micrograd_rs::nn::Trainer;
use micrograd_rs::nn::cross_entropy;

use micrograd_rs::{assert_close, assert_not_close};
use rand;

use rand_distr::{Distribution, Normal};

fn fake_data() -> (Vec<f64>, Vec<u8>) {
    let n = 1000;
    let mut rng = &mut rand::thread_rng();
    let mut data: Vec<f64> = Vec::with_capacity(2 * n);
    let mut labels: Vec<u8> = Vec::with_capacity(2 * n);

    // class 0
    let normal = Normal::new(0.0, 1.0).unwrap();
    data.extend((0..n).map(|_| normal.sample(rng)));
    labels.extend(vec![0u8; n]);

    // class 1
    let normal = Normal::new(2.0, 1.0).unwrap();
    data.extend((0..n).map(|_| normal.sample(rng)));
    labels.extend(vec![0u8; n]);

    (data, labels)
}

#[test]
fn test_tools() {
    assert_close!(1.0 + 2.0, 3.0);
    assert_close!(1e-3, 1e-3 + 1e-10);
    assert_not_close!(1e-3, 2e-3);
    assert_not_close!(1e-3, 1e-3 + 1e-5);
}

#[test]
fn test_add() {
    let a = Value::new(5.0);
    let b = Value::new(6.0);
    let c = a + b;
    assert_close!(c.data, 11.0);

    let a = Value::new(-5.0);
    let b = Value::new(6.0);
    let c = a + b;
    assert_close!(c.data, 1.0);
}

#[test]
fn test_sub() {
    let a = Value::new(5.0);
    let b = Value::new(6.0);
    let c = a - b;
    assert_close!(c.data, -1.0);

    let a = Value::new(5.0);
    let b = Value::new(-6.0);
    let c = a - b;
    assert_close!(c.data, 11.0);
}

#[test]
fn test_div() {
    let a = Value::new(5.0);
    let b = Value::new(6.0);
    let c = a / b;
    assert_close!(c.data, 5.0 / 6.0);

    let a = Value::new(5.0);
    let b = Value::new(-6.0);
    let c = a / b;
    assert_close!(c.data, 5.0 / -6.0);

    let a = Value::new(5.0);
    let b = Value::new(0.0);
    let c = a / b;
    assert_close!(c.data, 5.0 / 0.0);

    let a = Value::new(0.0);
    let b = Value::new(1.0);
    let c = a / b;
    assert_close!(c.data, 0.0 / 1.0);

    let a = Value::new(0.0);
    let b = Value::new(0.0);
    let c = a / b;
    assert_close!(c.data, 0.0 / 0.0);
}

// #[test]
// fn test_mlp() {
//     let (data, labels) = fake_data();
//     let mlp = MLP::new();
//     let trainer = Trainer::new(mlp, None).fit(&data, &labels);

//     let logits = mlp.forward(data);
//     let loss = cross_entropy(&labels, &logits);
// }
