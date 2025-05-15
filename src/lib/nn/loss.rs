use crate::engine::{DiscreteLabel, Value};
use core::f64;

/*
suppose x is a vector.

softmax(x)_i = exp(x_i) / sum(exp(x))

log_softmax(x)_i = log( exp(x_i) / sum(exp(x)) )
    = x_i - log(sum(exp(x))
    = x_i - log_sum_exp(x)


cross_entropy(pred_probs, true_probs) = true_probs * log(pred_probs)

logits = model(data)
pred_probs = softmax(logits)
log(pred_probs) = log(softmax(logits)) = log_softmax(logits)

cross_entropy(pred_probs, true_probs) = true_probs * log_softmax(logits)

*/

pub fn cross_entropy_single(label: DiscreteLabel, logits: &[Value]) -> Value {
    let log_probs = log_softmax(logits);
    let msg = format!("label must be in range [0, {}]", logits.len());
    -1.0 * log_probs.get(label).expect(&msg).clone()
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
    let offset = max_val(logits);
    let shifted_logits = logits.iter().map(|v| v - offset);

    shifted_logits.map(|v| v.exp()).fold(Value::from(0.0), |acc, val| acc + val).log() + offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_close;
    use crate::engine::{DiscreteLabel, Value};
    use crate::optim::{Optim, SGD};
    use anyhow::Result;
    use tch;
    use tch::nn::OptimizerConfig;
    use tch::Tensor;

    #[test]
    fn test_logsumexp() -> Result<()> {
        let logits = vec![Value::from(2.0), Value::from(2.0)];
        let lse1 = logsumexp(&logits);

        let logits_torch = Tensor::try_from(vec![2.0, 2.0])?;
        let lse2 = logits_torch.logsumexp(0, false);
        assert_eq!(lse1.data(), lse2.double_value(&[]));
        Ok(())
    }

    #[test]
    fn losses() -> Result<()> {
        // Try once with ours
        let logits = vec![Value::from(1.0), Value::from(1.0)];
        let y_true = 0 as DiscreteLabel;

        let loss = cross_entropy_single(y_true, &logits);
        let optim = SGD::new(logits.clone(), 1e-3);

        optim.zero_grad();
        loss.backward();
        optim.step();

        // Try once with torch

        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let logits_t = vs.root().ones("foo", &[2]);
        let y_true_t = Tensor::from(0i64);
        let loss_t = logits_t.cross_entropy_for_logits(&y_true_t);
        let mut optim_t = tch::nn::Sgd { momentum: 0.0, dampening: 0.0, wd: 0.0, nesterov: false }.build(&vs, 1e-3)?;

        optim_t.backward_step(&loss_t);
        assert_close!(loss.data(), loss_t.double_value(&[]));
        assert_close!(logits[0].grad().unwrap(), logits_t.get(0).grad().double_value(&[]));

        Ok(())
    }
}
