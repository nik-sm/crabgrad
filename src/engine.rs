use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};
use std::ptr;
use std::rc::Rc;

const PRECISION: usize = 3;

type BackwardType2 = fn(Value, Value) -> Value;

#[derive(Debug, Clone)]
pub struct Value {
    pub data: f64,
    pub grad: Option<f64>,
    backward_fn: Option<BackwardType2>,
    prev_nodes: Option<Vec<Rc<Value>>>,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self as *const Self).hash(state);
    }
}

impl Value {
    pub fn new(data: f64) -> Self {
        return Value { data, grad: None, backward_fn: None, prev_nodes: None };
    }

    pub fn to_string(&self) -> String {
        let grad = match self.grad {
            Some(x) => format!("{x:.PRECISION$}"),
            None => "None".to_string(),
        };
        format!("data: {:.PRECISION$}\ngrad: {:.PRECISION$}", self.data, grad)
    }

    pub fn pow(&self, n: f64) -> Value {
        Value::new(self.data.powf(n))
        /*

        def __pow__(self, other):
            assert isinstance(other, (int, float)), "only supporting int/float powers for now"
            out = Value(self.data**other, (self,), f'**{other}')

            def _backward():
                self.grad += (other * self.data**(other-1)) * out.grad
            out._backward = _backward

            return out

         */
    }

    pub fn backward(self: &mut Rc<Self>) {
        let mut visited = HashSet::new();
        let mut topo = Vec::new();

        fn build_topo(v: &Rc<Value>, visited: &mut HashSet<Rc<Value>>, topo: &mut Vec<Rc<Value>>) -> () {
            if !visited.contains(v) {
                visited.insert(Rc::clone(&v));
                if let Some(prev) = &v.prev_nodes {
                    for child in prev.iter().cloned() {
                        build_topo(&Rc::clone(&child), visited, topo);
                    }
                }
                topo.push(Rc::clone(v));
            }
        }

        build_topo(self, &mut visited, &mut topo);

        let root = Rc::get_mut(self).unwrap();
        root.grad = Some(1.0);

        for v in topo.iter().rev() {
            ();
        }
    }

    pub fn relu(&self) -> Value {
        Value::new(if self.data < 0.0 { 0.0 } else { self.data })
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
