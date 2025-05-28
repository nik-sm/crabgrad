use crabgrad::engine::{exp, to_vec, DiscreteLabel, Value};
use crabgrad::nn::loss::{log_softmax, nll_loss_single};
use crabgrad::optim::{Optim, SGD};

fn main() {
    // Try once with ours
    let logits = vec![Value::from(1.0), Value::from(1.0)];
    let y_true = 0 as DiscreteLabel;

    let probs_before = exp(&log_softmax(&logits));
    dbg!("before", to_vec(&logits), to_vec(&probs_before));

    let loss = nll_loss_single(y_true, &log_softmax(&logits));
    let mut optim = SGD::new(&logits, 1e-3);
    optim.zero_grad();
    loss.backward();
    optim.step();

    let probs_after = exp(&log_softmax(&logits));
    dbg!("after", to_vec(&logits), to_vec(&probs_after));
}
