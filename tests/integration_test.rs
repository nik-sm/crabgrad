mod utils;
use micrograd_rs::engine::Value;
use micrograd_rs::nn::cross_entropy;
use micrograd_rs::nn::MLP;

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

#[test]
fn test_mlp() {
    let mlp = MLP::new();
    let data = unimplemented!();
    let label = unimplemented!();
    let logits = mlp.forward(data);
    let loss = cross_entropy(label, logits);
}
