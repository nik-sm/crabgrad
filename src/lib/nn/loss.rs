use crate::engine::Value;
use core::f64;

pub fn cross_entropy_single(label: usize, logits: &[Value]) -> Value {
    let log_probs = log_softmax(logits);
    let msg = format!("label must be in range [0, {}]", logits.len());
    log_probs.get(label).expect(&msg).clone()
}

pub fn log_softmax(logits: &[Value]) -> Vec<Value> {
    let lse = logsumexp(logits);
    logits.iter().map(|l| l - lse.clone()).collect()
}

pub fn logsumexp(logits: &[Value]) -> Value {
    fn max_val(logits: &[Value]) -> f64 {
        let mut max_val = f64::NEG_INFINITY;
        for v in logits.iter().map(|v| v.data()) {
            if v.is_finite() && v > max_val {
                max_val = v
            }
        }
        if max_val == f64::NEG_INFINITY {
            max_val = 0.0;
        }
        max_val
    }

    // Subtract max value, which gives equivalent result but more numerically stable
    let shifted_logits = logits.iter().map(|v| v - max_val(logits));

    shifted_logits.map(|v| v.exp()).fold(Value::from(0.0), |acc, val| acc + val).log()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{DiscreteLabel, Value};
    use crate::optim::{Optim, SGD};

    #[test]
    fn losses() {
        let logits = vec![Value::from(1.0), Value::from(2.0)];
        let y_true = 0 as DiscreteLabel;

        let loss = cross_entropy_single(y_true, &logits);
        let optim = SGD::new(logits.clone(), 1.0);

        let target_value_before = logits.get(y_true).unwrap().data();
        dbg!("before", &logits);
        optim.zero_grad();
        loss.backward();
        optim.step();
        dbg!("after", &logits);

        let target_value_after = logits.get(y_true).unwrap().data();
        dbg!(target_value_before, target_value_after);
        // dbg!(loss);
        assert!(target_value_after > target_value_before);
        panic!("todo");
    }
}
