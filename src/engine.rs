use std::ops::{Add, Div, Mul, Sub};

const PRECISION: usize = 3;

#[derive(Debug)]
pub struct Value {
    pub data: f32,
    grad: Option<f32>,
    // backward_fn: Option<_>,
    // prev: Option<_>,
    // op: Option<_>,
}

impl Value {
    pub fn new(data: f32) -> Self {
        return Value { data, grad: None };
    }

    fn new_with_details(data: f32) -> Self {
        unimplemented!("")
    }

    pub fn to_string(&self) -> String {
        let data = self.data;
        let grad = match self.grad {
            None => String::from("None"),
            Some(x) => x.to_string(),
        };

        format!("{data:.PRECISION$} and {grad:.PRECISION$}",)
    }

    pub fn pow(&self, n: f32) -> Value {
        Value::new(self.data.powf(n))
    }
}

impl Add for Value {
    type Output = Self;

    fn add(self, rhs: Value) -> Self {
        Value::new(self.data + rhs.data)
    }
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, rhs: Value) -> Self {
        Value::new(self.data - rhs.data)
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Value) -> Self {
        Value::new(self.data * rhs.data)
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, rhs: Value) -> Self {
        Value::new(self.data / rhs.data)
    }
}
