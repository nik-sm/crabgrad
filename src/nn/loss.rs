use crate::engine::Value;
use core::f64;

pub fn cross_entropy(labels: &[Value], logits: &[Value]) -> Value {
    todo!()
}

pub fn cross_entropy_single(label: &Value, logits: &[Value]) -> Value {
    // Subtract max value, which gives equivalent result but more numerically stable
    let values: Vec<f64> = logits.iter().map(|v| v.data()).collect();
    let mut max_val = f64::NEG_INFINITY;
    for v in values {
        if v.is_finite() && v > max_val {
            max_val = v
        }
    }
    if max_val == f64::NEG_INFINITY {
        max_val = 0.0;
    }
    let shifted_logits = logits.iter().map(|v| v - max_val);

    // TODO - which terms should keep gradients?
    let numerator = shifted_logits.map(|v| v.exp());
    let denominator = &numerator.fold(0.0, |acc, val| acc + val.data());
    let probs = numerator.map(|val| val / denominator);
    todo!()
}

pub fn softmax(logits: Vec<Value>) -> Vec<Value> {
    todo!()
}
