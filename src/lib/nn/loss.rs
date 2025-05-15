use crate::engine::Value;
use core::f64;

pub fn cross_entropy(labels: &[Value], logits: &[Value]) -> Value {
    todo!()
}

pub fn cross_entropy_single(label: i64, logits: &[Value]) -> Value {
    let log_probs = log_softmax(logits);
    log_probs
        .iter()
        .nth(label as usize)
        .expect(format!("label must be in range [0, {}]", logits.len()).as_str())
        .clone()
}

pub fn log_softmax(logits: &[Value]) -> Vec<Value> {
    let lse = logsumexp(logits);
    logits.iter().map(|l| l - lse.clone()).collect()
}

pub fn logsumexp(logits: &[Value]) -> Value {
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

    shifted_logits.map(|v| v.exp()).fold(Value::from(0.0), |acc, val| acc + val).log()
}
