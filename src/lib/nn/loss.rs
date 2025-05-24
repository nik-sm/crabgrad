use crate::engine::{DiscreteLabel, Value};
use core::f64;

pub fn cross_entropy_single(label: DiscreteLabel, logits: &[Value]) -> Value {
    let log_probs = log_softmax(logits);
    let msg = format!("label must be in range [0, {}]", logits.len());
    -1.0 * log_probs.get(label).expect(&msg).clone()
}

pub fn log_softmax(logits: &[Value]) -> Vec<Value> {
    let lse = logsumexp(logits);
    logits.iter().map(|l| l - lse.clone()).collect()
}

pub fn max_val(values: &[Value]) -> f64 {
    let mut max_val = f64::NEG_INFINITY;
    for v in values.iter().map(|v| v.data()) {
        if v.is_finite() && v > max_val {
            max_val = v
        }
    }
    if max_val == f64::NEG_INFINITY {
        max_val = 0.0;
    }
    max_val
}

pub fn logsumexp(logits: &[Value]) -> Value {
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
    use tch::Tensor;
    use tch::nn::OptimizerConfig;

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
    fn losses1() -> Result<()> {
        // Try once with ours
        let logits = vec![Value::from(1.0), Value::from(1.0)];
        let y_true = 0 as DiscreteLabel;

        let loss = cross_entropy_single(y_true, &logits);
        let mut optim = SGD::new(logits.clone(), 1e-3);

        optim.zero_grad();
        loss.backward();
        optim.step();

        // Try once with torch
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let logits_t = vs.root().ones("foo", &[2]);
        let y_true_t = Tensor::from(0i64);
        let loss_t = logits_t.cross_entropy_for_logits(&y_true_t);
        let mut optim_t = tch::nn::sgd(0.0, 0.0, 0.0, false).build(&vs, 1e-3)?;
        optim_t.backward_step(&loss_t);

        assert_close!(loss.data(), loss_t.double_value(&[]));
        assert_close!(logits[0].grad().unwrap(), logits_t.grad().double_value(&[0]));
        assert_close!(logits[1].grad().unwrap(), logits_t.grad().double_value(&[1]));

        Ok(())
    }

    #[test]
    fn losses2() -> Result<()> {
        // Try once with ours
        let logits_a = vec![Value::from(1.0), Value::from(1.0)];
        let y_true_a = 0 as DiscreteLabel;

        let logits_b = vec![Value::from(3.0), Value::from(3.0)];
        let y_true_b = 1 as DiscreteLabel;

        let loss_a = cross_entropy_single(y_true_a, &logits_a);
        let loss_b = cross_entropy_single(y_true_b, &logits_b);

        let mut optim = SGD::new(logits_a.clone().into_iter().chain(logits_b.clone()).collect::<Vec<_>>(), 1e-3);
        let loss = Value::from(0.0) + loss_a + loss_b;

        optim.zero_grad();
        loss.backward();
        optim.step();

        // Try once with torch
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);

        let logits_a_t = Tensor::ones(2, (tch::Kind::Float, vs.device()));
        let logits_a_t = vs.root().add("bar", logits_a_t, true);
        let y_true_a_t = Tensor::from(0i64);

        let logits_b_t = Tensor::ones(2, (tch::Kind::Float, vs.device())) * 3.0;
        let logits_b_t = vs.root().add("bar", logits_b_t, true);
        let y_true_b_t = Tensor::from(1i64);

        let loss_a_t = logits_a_t.cross_entropy_for_logits(&y_true_a_t);
        let loss_b_t = logits_b_t.cross_entropy_for_logits(&y_true_b_t);

        let mut optim_t = tch::nn::sgd(0.0, 0.0, 0.0, false).build(&vs, 1e-3)?;
        optim_t.zero_grad();
        let loss_t = Tensor::from(0.0) + &loss_a_t + &loss_b_t;
        optim_t.backward_step(&loss_t);

        assert_close!(loss.data(), loss_t.double_value(&[]));

        // Check effects on logits_a
        dbg!(&logits_a);
        dbg!(&logits_b);
        dbg!(&logits_a_t);
        dbg!(&logits_b_t);
        assert_close!(logits_a[0].grad().unwrap(), logits_a_t.grad().double_value(&[0]));
        assert_close!(logits_a[1].grad().unwrap(), logits_a_t.grad().double_value(&[1]));

        // Check effects on logits_b
        assert_close!(logits_b[0].grad().unwrap(), logits_b_t.grad().double_value(&[0]));
        assert_close!(logits_b[1].grad().unwrap(), logits_b_t.grad().double_value(&[1]));

        Ok(())
    }
}
