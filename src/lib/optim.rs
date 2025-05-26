use crate::engine::{FloatDataScalar, Value};

pub trait Optim {
    fn zero_grad(&self);
    fn step(&mut self);
}

pub enum OptimType {
    SGD,
    ADAM,
}

pub struct SGD {
    parameters: Vec<Value>,
    lr: f64,
}

impl SGD {
    #[must_use]
    pub fn new(parameters: &[Value], lr: f64) -> Self {
        Self { parameters: parameters.to_owned(), lr }
    }
}

impl Optim for SGD {
    fn zero_grad(&self) {
        for p in &self.parameters {
            p.zero_grad();
        }
    }
    fn step(&mut self) {
        for p in &self.parameters {
            p.borrow_mut().data = self.lr.mul_add(-p.grad().expect("step without grad"), p.data());
        }
    }
}

pub struct AdamW {
    parameters: Vec<Value>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    momentum: Vec<FloatDataScalar>,
    velocity: Vec<FloatDataScalar>,
    time_step: i32,
}

impl AdamW {
    #[must_use]
    pub fn new(parameters: Vec<Value>, lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        let n = parameters.len();
        Self {
            parameters,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            momentum: vec![0.0; n],
            velocity: vec![0.0; n],
            time_step: 0,
        }
    }
}

impl Optim for AdamW {
    fn zero_grad(&self) {
        for p in &self.parameters {
            p.zero_grad();
        }
    }

    fn step(&mut self) {
        self.time_step += 1;
        for (idx, p) in self.parameters.iter().enumerate() {
            let grad = p.grad().expect("step without grad");
            let mom = self.beta1.mul_add(self.momentum[idx], (1.0 - self.beta1) * grad);
            self.momentum[idx] = mom;

            let vel = self.beta2.mul_add(self.velocity[idx], (1.0 - self.beta2) * grad.powi(2));
            self.velocity[idx] = vel;

            let first_moment = mom / (1.0 - self.beta1.powi(self.time_step));
            let second_moment = vel / (1.0 - self.beta2.powi(self.time_step));

            let old_val = p.data();
            p.borrow_mut().data = self
                .weight_decay
                .mul_add(old_val, self.lr.mul_add(-(first_moment / (second_moment.sqrt() + self.eps)), old_val));
        }
    }
}

// TODO - annoying to manually set all params. Either use builder pattern or Default::default
// impl Default for AdamW {
//     fn default() -> Self {
//         Self {
//             parameters: vec![],
//             lr: 1e-3,
//             beta1: 0.9,
//             beta2: 0.999,
//             eps: 1e-8,
//             weight_decay: 0.01,
//             momentum: vec![],
//             velocity: vec![],
//             time_step: 0,
//         }
//     }
// }
